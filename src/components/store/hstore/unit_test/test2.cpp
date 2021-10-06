/*
   Copyright [2017-2021] [IBM Corporation]
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

#include "pool_open.h"
#include "store_map.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wconversion"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <common/env.h>
#include <common/utils.h>
#include <api/components.h>
/* note: we do not include component source, only the API definition */
#include <api/kvstore_itf.h>

#include <cstddef>
#include <cstring>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

#if ! defined MAKE_SYNDROME_REPORT
#define MAKE_SYNDROME_REPORT 0
#endif

using namespace component;

// The fixture for testing class Foo.
class KVStore_test : public ::testing::Test {

	static constexpr std::size_t estimated_object_count_large = 64000000;
	/* More testing of table splits, at a performance cost */
	static constexpr std::size_t estimated_object_count_small = 1;
	static constexpr std::size_t many_count_target_large = 2000000;
	/* Shorter test: use when PMEM_IS_PMEM_FORCE=0 */
	static constexpr std::size_t many_count_target_small = 400;

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

	// Objects declared here can be used by all tests in the test case
	/* persistent memory if enabled at all, is simulated and not real */
	static bool pmem_simulated;
	/* persistent memory is effective (either real, indicated by no PMEM_IS_PMEM_FORCE or simulated by PMEM_IS_PMEM_FORCE 0 not 1 */
	static bool pmem_effective;
#if 0
	static component::IKVStore * _kvstore;
#endif

	static const std::size_t estimated_object_count;

	static constexpr unsigned many_key_length = 8;
	static constexpr unsigned many_value_length = 16;
	static constexpr char long_value[24] = "........" "........" ".......";
#if 0
	static std::vector<kv_t> kvv;
#endif
	static const std::size_t many_count_target;
	static std::size_t many_count_actual;

	std::string pool_name() const
	{
		return "pool/" + store_map::numa_zone() + "/test-" + store_map::impl->name;
	}
	static std::string debug_level()
	{
		return std::getenv("DEBUG") ? std::getenv("DEBUG") : "0";
	}
};

constexpr std::size_t KVStore_test::estimated_object_count_small;
constexpr std::size_t KVStore_test::estimated_object_count_large;
constexpr std::size_t KVStore_test::many_count_target_small;
constexpr std::size_t KVStore_test::many_count_target_large;
constexpr char KVStore_test::long_value[24];

bool KVStore_test::pmem_simulated = getenv("PMEM_IS_PMEM_FORCE");
bool KVStore_test::pmem_effective = ! getenv("PMEM_IS_PMEM_FORCE") || getenv("PMEM_IS_PMEM_FORCE") == std::string("0");
#if 0
component::IKVStore * KVStore_test::_kvstore;
#endif

const std::size_t KVStore_test::estimated_object_count =
/*
 * Any estimated count should work, subject to some size limit. A small count,
 * such as 0, causes more allocations and therefore a more rigorous test.
 */
#if 0
	pmem_simulated ? estimated_object_count_small : estimated_object_count_large
#else
	0 /* test resize */
#endif
	;

const std::size_t KVStore_test::many_count_target =
#if MAKE_SYNDROME_REPORT
	20
#else
	pmem_simulated ? many_count_target_small : many_count_target_large
#endif
	;

constexpr unsigned KVStore_test::many_key_length;
constexpr unsigned KVStore_test::many_value_length;

std::size_t KVStore_test::many_count_actual;
#if 0
std::vector<KVStore_test::kv_t> KVStore_test::kvv;
#endif

using kv_t = std::tuple<std::string, std::string>;

std::vector<kv_t> populate_many(const std::size_t many_count_target, unsigned many_key_length, unsigned many_value_length)
{
	std::vector<kv_t> kvv;
	std::mt19937_64 r0{};
	for ( auto i = 0UL; i != many_count_target; ++i )
	{
		auto ukey = r0();
		std::ostringstream s;
		s << std::hex << ukey;
		auto key = s.str();
		key.resize(many_key_length, '.');
		auto value = std::to_string(i);
		value.resize(many_value_length, '.');
		kvv.emplace_back(key, value);
	}
	return kvv;
}

/*
 * We would like to generate "crashes" with some reasonable frequency,
 * but not at every store. (Every store would be too slow, at least
 * when using mmap to simulate persistent store). We use a Fibonacci
 * series to produce crashes at decreasingly frequent intervals.
 *
 * _p0 and _p1 produce a Fibonacci series in perishable_count
 */
struct fib
{
	std::uint64_t _p0;
	std::uint64_t _p1;
	/* defined in perishable, but that definition not visibe to test cases */
	static constexpr auto use_syndrome = std::numeric_limits<std::uint64_t>::max();
	fib()
		: _p0(0)
		, _p1(1)
	{}
	std::uint64_t value() const
	{
#if MAKE_SYNDROME_REPORT
		return use_syndrome;
#else
		return _p0 + _p1;
#endif
	}

	std::uint64_t step()
	{
		auto v = value();
#if 0
/*
 * Crash after every other perishable operation. Guarantees progress, but is slow.
 */
#else
/*
 * Crash in decreasing frequency, by Fibonacci sequence.
 */
		_p0 = _p1;
		_p1 = v;
#endif
		return v;
	}
};

std::ostream &operator<<(std::ostream &o_, fib f_)
{
	if ( f_.value() == fib::use_syndrome )
	{
		o_ << "(syndrome)";
	}
	else
	{
		o_ << f_.value();
	}
	return o_;
}

TEST_F(KVStore_test, Instantiate)
{
	std::cerr
		<< "PMEM " << (pmem_simulated ? "simulated" : "not simluated")
		<< ", " << (pmem_effective ? "effective" : "not effective")
		<< "\n";
	/* create object instance through factory */
	/* This test only: use hstore-pe. the version compiled with simulated injection */
	auto link_library = "libcomponent-" + store_map::impl->name + "-pe.so";
	std::cerr << "link library: " << link_library << " mm plugin " << common::env_value<const char *>("MM_PLUGIN_PATH", "no_plugin_path") << "\n";
	component::IBase * comp = component::load_component(link_library,
															                        store_map::impl->factory_id);

	ASSERT_TRUE(comp);
	auto fact = make_itf_ref(static_cast<IKVStore_factory *>(comp->query_interface(IKVStore_factory::iid())));

	component::IKVStore * kvstore =
		fact->create(
			0
			, {
				{ +component::IKVStore_factory::k_name, "numa0"}
				, { +component::IKVStore_factory::k_dax_config, store_map::location }
				, { +component::IKVStore_factory::k_debug, debug_level() }
				, { +component::IKVStore_factory::k_mm_plugin_path, common::env_value<const char *>("MM_PLUGIN_PATH", "no_plugin_path") }
			}
		);

	ASSERT_NE(nullptr, kvstore);

	{
		try
		{
			kvstore->delete_pool(pool_name());
		}
		catch ( Exception & )
		{
		}
	}

	pool_temp pt(kvstore, pool_name(), GB(15UL), 0, estimated_object_count);
	ASSERT_NE(+IKVStore::POOL_ERROR, pt.id());

	std::vector<kv_t> kvv = populate_many(many_count_target, many_key_length, many_value_length);

	{
		/* We will try the inserts many times, as the perishable timer will abort all but the last attempt */
		bool finished = false;

		kvstore->debug(0, 0 /* enable */, 0);
		for ( fib perishable_count{}
			; ! finished
			; perishable_count.step()
			)
		{
			kvstore->debug(0, 1 /* reset */, perishable_count.value());
			kvstore->debug(0, 0 /* enable */, true);

			unsigned extant_count = 0;
			unsigned succeed_count = 0;
			unsigned fail_count = 0;
			try
			{
				pool_open p(kvstore, pool_name());

				for ( auto &kv : kvv )
				{
					const auto &key = std::get<0>(kv);
					const auto &value = std::get<1>(kv);
					void * old_value = nullptr;
					size_t old_value_len = 0;
					if ( S_OK == kvstore->get(p.id(), key, old_value, old_value_len) )
					{
						kvstore->free_memory(old_value);
						++extant_count;
					}
					else
					{
						auto r = kvstore->put(p.id(), key, value.c_str(), value.length());
						EXPECT_EQ(S_OK, r);
						if ( r == S_OK )
						{
							++succeed_count;
						}
						else
						{
							++fail_count;
						}
					}
				}
				EXPECT_EQ(many_count_target, extant_count + succeed_count + fail_count);
				many_count_actual = extant_count + succeed_count;
				finished = true;
				/* Done with forcing crashes */
				kvstore->debug(0, 0 /* enable */, false);
				std::cerr << __func__ << " Final put pass " << perishable_count << " exists " << extant_count << " inserts " << succeed_count << " total " << many_count_actual << "\n";
			}
			catch ( const std::runtime_error &e )
			{
				if ( e.what() != std::string("perishable timer expired") ) { throw; }
				std::cerr << __func__ << " Perishable pass " << perishable_count << " exists " << extant_count << " inserts " << succeed_count << " total " << many_count_actual << "\n";
			}
		}
	}

	{
		if ( pmem_effective )
		{
			pool_open p(kvstore, pool_name());
			ASSERT_LT(0, int64_t(p.id()));
			auto count = kvstore->count(p.id());
			{
				/* count should be close to PutMany many_count_actual; duplicate keys are the difference */
				EXPECT_LE(double(many_count_actual) * 99. / 100., count);
			}
			{
				std::size_t mismatch_count = 0;
				for ( auto &kv : kvv )
				{
					const auto &key = std::get<0>(kv);
					const auto &ev = std::get<1>(kv);
					void * value = nullptr;
					size_t value_len = 0;
					auto r = kvstore->get(p.id(), key, value, value_len);
					EXPECT_EQ(S_OK, r);
					if ( S_OK == r )
					{
						EXPECT_EQ(ev.size(), value_len);
						mismatch_count += ( ev.size() != value_len || 0 != memcmp(ev.data(), value, ev.size()) );
						kvstore->free_memory(value);
					}
				}
				/* We do not know exactly now many mismatches (caused by duplicates) to expect,
				 * because "extant_count" counts both extant items due to duplicate keys in the
				 * population arrays and extant items due to restarts.
				 * But it should be a small fraction of the total number of keys
				 */
				EXPECT_GT(many_count_target / 100, mismatch_count);
			}
		}
	}

	{
		/* We will try the inserts many times, as the perishable timer will abort all but the last attempt */
		bool finished = false;

		kvstore->debug(0, 0 /* enable */, 0);
		for ( fib perishable_count{}
			; ! finished
			; perishable_count.step()
			)
		{
			kvstore->debug(0, 1 /* reset */, perishable_count.value());
			kvstore->debug(0, 0 /* enable */, true);

			unsigned extant_count = 0;
			unsigned succeed_count = 0;
			unsigned fail_count = 0;
			try
			{
				pool_open p(kvstore, pool_name());

				for ( auto &kv : kvv )
				{
					/* Long enough to force a move to out-of-line storage */
					const auto &key = std::get<0>(kv);
					const auto &value = std::get<1>(kv);
					const auto update_value = value + ((key[0] & 1) ? "X" : "") + ((key[0] & 2) ? long_value : "");

					void * extant_value = nullptr;
					size_t extant_value_len = 0;
					auto gr = kvstore->get(p.id(), key, extant_value, extant_value_len);
					ASSERT_EQ(S_OK, gr);
					if ( extant_value_len == update_value.size() && 0 == std::memcmp(extant_value, update_value.c_str(), extant_value_len) )
					{
						++extant_count;
					}
					else
					{
						auto r = kvstore->put(p.id(), key, update_value.c_str(), update_value.length());
						EXPECT_EQ(S_OK, r);
						if ( r == S_OK )
						{
							++succeed_count;
						}
						else
						{
							++fail_count;
						}
					}
					kvstore->free_memory(extant_value);
				}
				/* Due to forced crashes we may never see a success or failure, but all new values should exist */
				EXPECT_EQ(many_count_actual, extant_count + succeed_count);
				finished = true;
				/* Done with forcing crashes */
				kvstore->debug(0, 0 /* enable */, false);
				std::cerr << __func__ << " Final put pass " << perishable_count << " inserts " << succeed_count << " total " << many_count_actual << "\n";
			}
			catch ( const std::runtime_error &e )
			{
				if ( e.what() != std::string("perishable timer expired") ) { throw; }
				std::cerr << __func__ << " Perishable pass " << perishable_count << " inserts " << succeed_count << " total " << many_count_actual << "\n";
			}
		}
	}

	{
		kvstore->debug(0, 0 /* enable */, 0);
		ASSERT_TRUE(kvstore);
		if ( pmem_effective )
		{
			pool_open p(kvstore, pool_name());
			ASSERT_NE(+IKVStore::POOL_ERROR, p.id());
			auto count = kvstore->count(p.id());
			{
				/* count should be close to PutMany many_count_actual; duplicate keys are the difference */
				EXPECT_LE(many_count_actual * 99 / 100, double(count));
			}
			{
				std::size_t mismatch_count = 0;
				for ( auto &kv : kvv )
				{
					const auto &key = std::get<0>(kv);
					const auto &ev = std::get<1>(kv);
					const auto update_ev = ev + ((key[0] & 1) ? "X" : "") + ((key[0] & 2) ? long_value : "");
					void * value = nullptr;
					size_t value_len = 0;
					auto r = kvstore->get(p.id(), key, value, value_len);
					EXPECT_EQ(S_OK, r);
					if ( S_OK == r )
					{
						if ( update_ev.size() != value_len )
						{
							std::cerr << "Length mismatch, key length " << key.size() << " key " << key << " expected value " << update_ev << " actual value " << std::string(static_cast<char *>(value), value_len) << " key[0] " << key[0] << " selector " << (key[0] & 1) << "\n";
						}
						EXPECT_EQ(update_ev.size(), value_len);
						mismatch_count += ( update_ev.size() != value_len || 0 != memcmp(update_ev.data(), value, update_ev.size()) );
						kvstore->free_memory(value);
					}
				}
				/* We do not know exactly now many mismatches (caused by duplicates) to expcect,
				 * because "extant_count" counts both extant items due to duplicate keys in the
				 * population arrays and extant items due to restarts.
				 * But it should be a small fraction of the total number of keys
				 */
				EXPECT_GT(many_count_target / 100, mismatch_count);
			}
		}
	}

	{
		bool finished = false;
		auto erase_count = 0;

		kvstore->debug(0, 0 /* enable */, 0);
		for ( fib perishable_count{}
			; ! finished
			; perishable_count.step()
			)
		{
			kvstore->debug(0, 1 /* reset */, perishable_count.value() );
			kvstore->debug(0, 0 /* enable */, true);
			try
			{
				pool_open p(kvstore, pool_name());
				ASSERT_LT(0, int64_t(p.id()));
				{
					for ( auto &kv : kvv )
					{
						const auto &key = std::get<0>(kv);
						auto r = kvstore->erase(p.id(), key);
						if ( r == S_OK )
						{
							++erase_count;
						}
					}
					auto count = kvstore->count(p.id());
					EXPECT_EQ(0U, count);
					finished = true;
				}
			}
			catch ( const std::runtime_error &e )
			{
				if ( e.what() != std::string("perishable timer expired") ) { throw; }
				std::cerr << __func__ << " Perishable pass " << perishable_count << " erasures " << erase_count << " total " << many_count_actual << "\n";
			}
		}
	}
}

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	auto r = RUN_ALL_TESTS();

	return r;
}
