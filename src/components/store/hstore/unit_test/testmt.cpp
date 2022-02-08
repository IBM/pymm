/*
   Copyright [2017-2020] [IBM Corporation]
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "memo_lock.h"
#include "pool_open.h"
#include "store_map.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <api/components.h>
/* note: we do not include component source, only the API definition */
#include <api/kvstore_itf.h>
#include <common/byte_span.h>
#include <common/env.h>
#include <common/moveable_ptr.h>
#include <common/string_view.h>
#include <common/utils.h> /* MiB, GiB */
#include <nupm/region_descriptor.h>

#include <algorithm>
#include <chrono>
#include <cstring> /* memcmp */
#include <future>
#include <set>
#include <sstream>
#include <string>

/*
 * For debug, XLOG should assemble its arguments, add a terminating \n,
 * and send them to cerr.
 */
#define XLOG(a, ...) do {} while(0)

using namespace component;

namespace {

// The fixture for testing class KVStore_test.
class KVStore_test : public ::testing::Test
{
 protected:

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  virtual void SetUp() {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  virtual void TearDown() {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }
};

static std::string debug_level()
{
	return std::getenv("DEBUG") ? std::getenv("DEBUG") : "0";
}

Itf_ref<IKVStore_factory> load_component()
{
	/* load factory */
	auto link_library = "libcomponent-" + store_map::impl->name + ".so";
	auto comp = component::load_component(link_library, store_map::impl->factory_id);
	return component::make_itf_ref(static_cast<IKVStore_factory *>(comp ? comp->query_interface(IKVStore_factory::iid()) : nullptr));
}

component::IKVStore *make_store(Itf_ref<IKVStore_factory> &factory)
{
	/* create object instance through factory */
	/* numa node 0 */
	return
		factory->create(
			0
			, {
				{ +component::IKVStore_factory::k_name, "numa0"}
				, { +component::IKVStore_factory::k_dax_config, store_map::location }
				, { +component::IKVStore_factory::k_debug, debug_level() }
				, { +component::IKVStore_factory::k_mm_plugin_path, common::env_value<const char *>("MM_PLUGIN_PATH", "no_plugin_path") }
			}
		);
}

void incr_n_times(
	component::IKVStore *kvstore_, const std::string poolname_, const std::string data_key_, unsigned n_, unsigned total_, const std::string th_, const std::string descr_
)
{
	using namespace component;
	pool_open pool(kvstore_, poolname_);
	ASSERT_NE(+IKVStore::POOL_ERROR, pool.id());

#if 0
	timer t(
		[&descr, n_] (timer::duration_t d) {
		auto seconds = std::chrono::duration<double>(d).count();
		std::cerr << descr << " " << double(n_) / seconds << " per second\n";
			}
		);
#endif
	const unsigned align = 0;
	for ( ; n_ != 0; --n_ )
	{
		void *data;
		unsigned long data_size = 0;
		exclusive_lock lk(kvstore_, pool, th_);
		status_t r = kvstore_->lock(pool.id(), std::string(data_key_), IKVStore::STORE_LOCK_WRITE, data, data_size, align, lk.k);
		for ( auto delay = std::chrono::milliseconds(1); r == E_LOCKED; delay *= 2 )
		{
			std::this_thread::sleep_for(delay);
			r = kvstore_->lock(pool.id(), std::string(data_key_), IKVStore::STORE_LOCK_WRITE, data, data_size, align, lk.k);
		}
		if ( r != S_OK )
		{
			std::ostringstream e;
			e << __func__ << " " << descr_ << " failed code " << r << " key " << data_key_ << " data_size " << data_size << "\n";
			std::cerr << e.str();
		} 
		XLOG("(", n_, ") ", th_, " exc");

		ASSERT_EQ(S_OK, r);

		std::string s(static_cast<char *>(data), data_size);
		unsigned value;
		std::istringstream is(s);
		is >> value;
		++value;
		XLOG("(", n_, ") ", th_, " ", value, "<-", value-1);
		std::ostringstream os;
		os << std::setfill('0') << std::setw(10) << value;
		const char *cdata = os.str().data();
		std::copy(cdata, cdata+10, static_cast<char *>(data));
		ASSERT_EQ(data_size, os.str().size());
		ASSERT_EQ(S_OK, r);
	}

	unsigned current;
	do {
		{
			void *data;
			unsigned long data_size = 0;
			shared_lock lk(kvstore_, pool, th_);
			status_t r = kvstore_->lock(pool.id(), std::string(data_key_), IKVStore::STORE_LOCK_READ, data, data_size, align, lk.k);
			for ( auto delay = std::chrono::milliseconds(1); r == E_LOCKED; delay *= 2 )
			{
			std::this_thread::sleep_for(delay);
				r = kvstore_->lock(pool.id(), std::string(data_key_), IKVStore::STORE_LOCK_READ, data, data_size, align, lk.k);
			}
			ASSERT_EQ(S_OK, r);
			XLOG( th_, " shr");
			std::string s(static_cast<char *>(data), data_size);
			std::istringstream is(s);
			is >> current;
			XLOG(th_, " current ", current, " of ", total_);
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(1200));
	} while ( current != total_ );
}

void incr_locked(
	component::IKVStore *kvstore_
	, const std::string poolname_
	, const common::string_view data_key_
	, const unsigned thread_count_
	, const unsigned incr_count_
	, const std::string &descr_
)
{
	std::vector<std::future<void>> v;
#if 0
	/* profile disabled, run with env var PROFILE=1 to enable */
	common::profiler pr("incr_locked-" + descr + "-cpu-" + store_map::impl->name + ".profile", false);
#endif
	for ( auto i = thread_count_; i != 0; --i )
	{
		v.emplace_back(std::async(std::launch::async, incr_n_times, kvstore_, std::string(poolname_), std::string(data_key_), incr_count_, incr_count_ * thread_count_, std::string(1, char('A' + i-1)), descr_));
	}
	for ( auto &e : v ) { e.get(); }
}

TEST_F(KVStore_test, ThreadedIncrement)
{
	auto factory = load_component();
	ASSERT_NE(nullptr, factory);
	std::unique_ptr<component::IKVStore> kvstore(make_store(factory));
	ASSERT_NE(nullptr, kvstore);

	using namespace component;

	const std::string poolname = "LockedPutGetOperations";

	std::string key0   = "key0";
	pool_temp pool(
		kvstore.get(),
		poolname, MiB(32), /* size */
		0, /* flags */
		100             /* obj count */
	);

	ASSERT_NE(+IKVStore::POOL_ERROR, pool.id());

	{
		std::string value0 = "0000000000";
		ASSERT_EQ(S_OK, kvstore->put(pool.id(), key0, value0.data(), value0.size()));
	}

	incr_locked(kvstore.get(), poolname, key0, 4, 25, poolname);

	{
		std::string value100 = "0000000100";
		void *value1 = nullptr;
		std::size_t size = 10;
		ASSERT_EQ(S_OK, kvstore->get(pool.id(), key0, value1, size));
		ASSERT_EQ(10, size);
		ASSERT_EQ(0, std::memcmp(value100.data(), value1, 10));
		kvstore->free_memory(value1);
	}
}

} // namespace

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
