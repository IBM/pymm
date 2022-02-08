/*
   Copyright [2017-2022] [IBM Corporation]
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

#include "make_kvstore.h"
#include "kv_lock.h"
#include "pool_instance.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <common/json.h>
#include <common/logging.h> /* format */
#include <common/str_utils.h>
#include <common/utils.h>
#include <api/kvstore_itf.h>
#include <boost/program_options.hpp>
#include <gsl/pointers>
#include <nupm/region_descriptor.h>
#include <cstddef> /* size_t */
#include <iostream> /* cerr, cout */
#include <random>
#include <string>

#define ASSERT_OK(X) ASSERT_EQ(S_OK, (X))

static std::string store;
static custom_store *cs;
static component::IKVStore_factory::map_create mc;
static std::size_t many_count_proposed;

/*
 * Ideally, these error values should be expected wherever they might occur:
 *    E_KEY_EXISTS     = E_ERROR_BASE - 1 (put, put_direct)
 *    E_KEY_NOT_FOUND  = E_ERROR_BASE - 2, (resive_value, get, get_direct, get_attributes, swap_keys, lock)
 *    E_POOL_NOT_FOUND = E_ERROR_BASE - 3, (close_pool, get_[pool_]regions, grow_pool, put, put_direct, get, get_direct, get_attribute, swap_keys, set_attribute, map, map_keys, allocate_pool_memory, free_pool_memory, flush_pool_memory)
 *    E_BAD_ALIGNMENT  = E_ERROR_BASE - 4, (resize_value)
 *    E_TOO_LARGE      = E_ERROR_BASE - 5, (resize_value, lock)
 *    E_ALREADY_OPEN   = E_ERROR_BASE - 6, (close_pool, delete_pool)
*/

std::string pool_name()
{
	return "pool/test-" + store;
}

namespace
{
	/* A pool instance whose size is adapted depending in available memory.
	 * The variable size portion is recorded in "variable."
	 */
	struct pool_instance_sized
		: pool_instance
	{
	private:
		std::size_t _variable;
	public:
		template <typename ... Args>
			pool_instance_sized(
				gsl::not_null<std::shared_ptr<IKVStore>> kvstore_
				, common::string_view name_
				, std::size_t fixed_
				, std::size_t variable_
				, Args && ... args_
			)
				: pool_instance(kvstore_, name_, fixed_ + variable_, std::forward<Args>(args_) ...)
				, _variable(variable_)
			{}
		std::size_t variable() const { return _variable; }
	};

	/* a view into a vector, limited to the first limit_ elements */
	template <typename T>
		struct limited_size_view
		{
		private:
			const T &_t;
			typename T::size_type _size;
		public:
			limited_size_view(const T &t_, typename T::size_type limit_)
				: _t(t_)
				, _size(std::min(limit_, _t.size()))
			{}
			auto begin() const { return _t.begin(); }
			auto end() const { return begin() + _size; }
			auto cbegin() const { return _t.begin(); }
			auto cend() const { return begin() + _size; }
			auto rbegin() const { return rend() - _size; }
			auto rend() const { return _t.rend(); }
			auto crbegin() const { return crend() - _size; }
			auto crend() const { return _t.crend(); }
			auto size() const { return _size; }
		};

// The fixture for testing class Foo.
class KVStore_test
	: public ::testing::Test
{
	static bool numa_notified;
public:
	using IKVStore = component::IKVStore;
	using IKVStore_factory = component::IKVStore_factory;
	using string_view = common::string_view;
	using kv_t = std::tuple<std::string, std::string>;
protected:
	static const std::string test1_pool;
	static const std::string iterator_test_pool;
	static const std::string timestamp_test_pool;
	static const std::string single_key;
	static const std::string missing_key;
	static constexpr std::size_t single_value_size = MiB(8);
	static const std::string single_value_updated_different_size;
	static const std::string single_value_updated3;

	static constexpr std::size_t single_count = 1U;
	static constexpr unsigned get_expand = 2;
	/* AVL allocator expects 128MiB region for each slab */
	static constexpr std::size_t avl_minimum = MiB(320);

	/* More testing of table splits, at a performance cost */
	static constexpr std::size_t estimated_object_count_small = 1;
	static constexpr char hello[] = "Hello world!";

	// Objects declared here can be used by all tests in the test case
	gsl::not_null<std::shared_ptr<IKVStore>> _kvstore;
	/*
	 * The number of elements we would like for "many" operations".
	 * May be reduced to what the pool can handle
	 */
	std::size_t many_count_target_proposed;
	const std::vector<kv_t> kvv;

	static constexpr std::size_t many_key_length = 8;
	static constexpr std::size_t many_value_length = 16;

	const std::string single_value;
	std::string single_value_updated_same_size;

	KVStore_test()
		: _kvstore(make_kvstore(store, cs, mc))
		, many_count_target_proposed(many_count_proposed)
		, kvv(populate_many(many_count_target_proposed, many_key_length, many_value_length))
		, single_value(hello + std::string(single_value_size - (strlen(hello)), '\0'))
		, single_value_updated_same_size("Jello world!")
	{
		assert(single_value.size() == single_value_size);
		if ( ! mc.count(+IKVStore_factory::k_numa_nodes) == 0 && ! numa_notified )
		{
			std::cerr << "If using mmap memory. Run with --numa-nodes==<numa_nodes> to allocate from numa nodes\n";
			numa_notified = true;
		}
	}

	template <typename ... Args>
		auto open_pool(string_view pool_name_, Args && ... args_) -> pool_opened
		{
			return pool_opened(_kvstore, _kvstore->open_pool(std::string(pool_name_), std::forward<Args>(args_) ...));
		}

	auto create_pool(common::string_view name_)
	{
		return pool_instance(_kvstore, name_);
	}

	template <typename ... Args>
		auto create_pool(common::string_view name_, std::size_t size_, Args && ... args_)
		{
			return pool_instance(_kvstore, name_, std::max(size_, avl_minimum), std::forward<Args>(args_) ...);
		}

	template <typename ... Args>
		auto create_pool_sized(
			common::string_view name_
			, std::size_t fixed
			, std::size_t variable
			, Args & ... args_
		) -> pool_instance_sized
		{
			fixed = std::max(fixed, avl_minimum);
			auto p = pool_instance_sized(_kvstore, name_, fixed, variable, args_ ...);
			while ( p.handle() == IKVStore::POOL_ERROR && 0 != variable )
			{
				variable = variable / 2;
				p = std::move(pool_instance_sized(_kvstore, name_, fixed, variable, args_ ...));
			}
			return p;
		}
	/*
	 * Make a pool when the pool may
	 * need to share itself, as with a lock
	 */
	template <typename ... Args>
		auto create_pool_shared(common::string_view name_, std::size_t size_, Args && ... args_)
		{
			return std::make_shared<pool_instance>(_kvstore, name_, std::max(size_, avl_minimum), std::forward<Args>(args_) ...);
		}

	template <typename ... Args>
		auto create_pool_sized_shared(
			common::string_view name_
			, std::size_t fixed
			, std::size_t variable
			, Args & ... args
		)
		{
			fixed = std::max(fixed, avl_minimum);
			auto sp = std::make_shared<pool_instance_sized>(_kvstore, name_, fixed, variable, args ...);
			while ( sp->handle() == IKVStore::POOL_ERROR && 0 != variable )
			{
				variable = variable / 2;
				sp = std::make_shared<pool_instance_sized>(_kvstore, name_, fixed, variable, args ...);
			}
			return sp;
		}

		constexpr auto mc_count_to_size(std::size_t count_)
		{
			/* count of elements multiplied
			 *  - by 64U for bucket size,
			 *  - by 3U to account for 40% table density at expansion (typical),
			 *  - by 2U to account for worst-case due to doubling strategy for increasing bucket array size
			 * required size multiplied
			 *  - by 8U to account for current AVL_LB allocator alignment requirements
			 */
			return count_ * 64U * 3U * 2U * 8U;
		}

		constexpr auto mc_size_to_count(std::size_t size_)
		{
			return size_ / (64U * 3U * 2U * 8U);
		}

		constexpr auto sv_count_to_size(std::size_t count_)
		{
			return 4U * count_ * 8U;
		}
private:

	static std::vector<kv_t> populate_many(std::size_t mct, std::size_t mkl, std::size_t mvl)
	{
		std::vector<kv_t> r;
		std::mt19937_64 r0{};
		for ( auto i = std::size_t(); i != mct; ++i )
		{
			auto key = common::format("{:x}", r0());
			key.resize(mkl, '.');
			auto value = std::to_string(i);
			value.resize(mvl, '.');
			r.emplace_back(key, value);
		}
		return r;
	}
};

constexpr std::size_t KVStore_test::estimated_object_count_small;
constexpr std::size_t KVStore_test::many_key_length;
constexpr std::size_t KVStore_test::many_value_length;
constexpr std::size_t KVStore_test::single_count;
constexpr unsigned KVStore_test::get_expand;
constexpr std::size_t KVStore_test::avl_minimum;
constexpr std::size_t KVStore_test::single_value_size;

bool KVStore_test::numa_notified = false;
const std::string KVStore_test::test1_pool = "test1.pool";
const std::string KVStore_test::timestamp_test_pool = "timestamp-test.pool";
const std::string KVStore_test::iterator_test_pool = "iterator-test.pool";

/* Keys 23-byte or fewer are stored inline. Provide one longer to force allocation */
const std::string KVStore_test::single_key = "MySingleKeyLongEnoughToForceAllocation";
const std::string KVStore_test::missing_key = "KeyNeverInserted";
const std::string KVStore_test::single_value_updated_different_size = "Hello world!";
const std::string KVStore_test::single_value_updated3 = "WeXYZ world!";


void put2(pool_opened & pool)
{
	std::string key = "MyKey";
	std::string key2 = "MyKey2";
	std::string value = "Hello world!";
	//  value.resize(value.length()+1); /* append \0 */
	value.resize(KB(8));

	pool.put(key, value.c_str(), value.length());
	pool.put(key2, value.c_str(), value.length());
}

void show_percent_used(pool_opened & pool)
{
	using IKVStore = component::IKVStore;
	std::vector<uint64_t> attr;
	auto r = pool.get_attribute(IKVStore::PERCENT_USED, attr, nullptr);
	EXPECT_EQ(cs->rc_percent_used(), r);
	if ( S_OK == r )
	{
		EXPECT_GE(100, attr[0]);
		PINF("Percent used %zu", attr[0]);
	}
}

/*
 * Following tests were copied from test1 in mapstore unit_test
 */

TEST_F(KVStore_test, OpenPool)
{
	auto pool = create_pool(test1_pool);
	ASSERT_LT(component::IKVStore::POOL_ERROR, pool.handle());
}

TEST_F(KVStore_test, BasicPut)
{
	auto pool = create_pool(test1_pool);
	ASSERT_LT(component::IKVStore::POOL_ERROR, pool.handle());
	put2(pool);
}

TEST_F(KVStore_test, BasicGet)
{
	auto pool = create_pool(test1_pool);
	ASSERT_LT(component::IKVStore::POOL_ERROR, pool.handle());
	put2(pool);
	std::string key = "MyKey";

	void * value = nullptr;
	size_t value_len = 0;
	pool.get(key, value, value_len);
	PINF("Value=(%.50s) %lu", static_cast<const char *>(value), value_len);

	ASSERT_NE(nullptr, value);
	ASSERT_EQ(KB(8), value_len);
	pool.free_memory(value);

	value = nullptr;
	value_len = 0;
	pool.get(key, value, value_len);
	PINF("Repeat Value=(%.50s) %lu", static_cast<const char *>(value), value_len);
	auto count = pool.count();
	PINF("Count = %ld", count);
	ASSERT_EQ(2, count);
	ASSERT_NE(nullptr, value);
	ASSERT_EQ(KB(8), value_len);
	pool.free_memory(value);
}

TEST_F(KVStore_test, BasicMap1)
{
	auto pool = create_pool(test1_pool);
	put2(pool);
	pool.map(
		[](const void * key,
	                 const size_t key_len,
	                 const void * value,
	                 const size_t value_len) -> int
		{
			FINF("key:({}) value({})"
				, string_view(static_cast<const char *>(key), key_len)
				, string_view(static_cast<const char *>(value), value_len)
			);
			return 0;
		}
	);
}

TEST_F(KVStore_test, ValueResize)
{
	auto pool = create_pool(test1_pool);
	put2(pool);
	pool.map(
		[](
			const void * key,
			size_t key_len,
			const void * value,
			size_t value_len) -> int
		{
			FINF("key:({}) value({}-{})"
				, string_view(static_cast<const char *>(key), key_len)
				, string_view(static_cast<const char *>(value), value_len)
				, value_len
			);
			return 0;
		}
	);

	ASSERT_OK(pool.resize_value("MyKey", KB(16), 8));

	pool.map(
		[](
			const void * key,
			size_t key_len,
			const void * value,
			size_t value_len) -> int
		{
			FINF("key:({}) value({}-{})"
				, string_view(static_cast<const char *>(key), key_len)
				, string_view(static_cast<const char *>(value), value_len)
				, value_len
			);
	                return 0;
		}
	);

}

TEST_F(KVStore_test, BasicRemove)
{
	auto pool = create_pool(test1_pool);
	put2(pool);
	pool.erase("MyKey");
}

TEST_F(KVStore_test, ClosePool1)
{
	auto pool = create_pool(test1_pool);
	EXPECT_EQ(S_OK, pool.close());
}

TEST_F(KVStore_test, ReopenPool)
{
	auto pool = create_pool(test1_pool);
	FLOG("re-opened pool: {}", reinterpret_cast<const void *>(pool.handle()));

	pool.close();
}

TEST_F(KVStore_test, Timestamps)
{
	auto pool = create_pool(timestamp_test_pool, MB(32));
	/* if timestamping is enabled */
	if(_kvstore->get_capability(IKVStore::Capability::WRITE_TIMESTAMPS)) {

		auto now = common::epoch_now();

		for(unsigned i=0;i<10;i++) {
			auto value = common::random_string(16);
			auto key = common::random_string(8);
			FLOG("adding key-value pair ({})", key);
			pool.put(key, value.c_str(), value.size());
			sleep(2);
		}

		pool.map(
			[](const void* key,
	                         const size_t key_len,
	                         const void* value,
	                         const size_t value_len,
	                         const common::tsc_time_t timestamp) -> bool
			{
				(void)value; // unused
				(void)value_len; // unused
				FLOG("Timestamped record: {} @ {}"
					, string_view(static_cast<const char *>(key), key_len)
					, timestamp.raw()
				);
				return true;
			}
			, 0, 0
		);

		PLOG("After 5 seconds");
		pool.map(
			[](const void* key,
	                         size_t key_len,
	                         const void* value,
	                         size_t value_len,
	                         common::tsc_time_t timestamp) -> bool
			{
				(void)value; // unused
				(void)value_len; // unused
				FLOG("After 5 Timestamped record: {} @ {}"
					, string_view(static_cast<const char *>(key), key_len)
					, timestamp.raw()
				);
				return true;
			}
			, now.add_seconds(5)
			, common::epoch_time_t{0,0}
		);
	}

	PLOG("Closing pool.");
	EXPECT_NE(IKVStore::POOL_ERROR, pool.handle());
	EXPECT_EQ(S_OK, pool.close());
}

/* original copied from mapstore */
TEST_F(KVStore_test, Timestamps2)
{
	auto pool = create_pool("timestamp-test.pool", MiB(32), IKVStore::FLAGS_CREATE_ONLY);

	/* if timestamping is enabled */
	if ( _kvstore->get_capability(IKVStore::Capability::WRITE_TIMESTAMPS) )
	{
		auto t0 = common::epoch_now();
		unsigned delay = 2;

		std::vector<std::string> keys;
		for ( unsigned i=0; i<10; ++i )
		{
			keys.push_back(common::random_string(8));
		}

		for ( unsigned i=0; i != keys.size(); ++i )
		{
			auto value = common::random_string(16);
			PLOG("adding key-value pair (%s)", keys[i].c_str());
			pool.put(keys[i], value.c_str(), value.size());
			sleep(delay);
		}

		std::size_t key_count = 0;

		/* All keys should be listed */
		pool.map(
			[&key_count](
				const void* key
				, const size_t key_len
				, const void* // value
				, const size_t // value_len
				, const common::tsc_time_t timestamp
			) -> bool
			{
				++key_count;
				PLOG("Timestamped record: %.*s @ %lu", int(key_len), static_cast<const char *>(key), timestamp.raw());
				return true;
			}
			, 0
			, 0
		);
		EXPECT_EQ(keys.size(), key_count);

		unsigned wait = 5;
		auto t0_plus_wait = t0;
		t0_plus_wait.add_seconds(wait);

		/* First two keys should not be listed */
		PLOG("After %u seconds", wait);
		key_count = 0;
		pool.map(
			[&key_count, wait](
				const void* key
				, const size_t key_len
				, const void* // value
				, const size_t // value_len
				, const common::tsc_time_t timestamp
			) -> bool
			{
				++key_count;
				std::ostringstream ts;
				ts << timestamp;
				PLOG("After %u Timestamped record: %.*s @ %s", wait
					, int(key_len), static_cast<const char *>(key), ts.str().c_str()
				);
				return true;
			}
			/* A time_point, expressed as seconds past epoch */
			, t0_plus_wait
			, 0
		);
		EXPECT_EQ(keys.size() - unsigned(std::ceil(double(wait)/double(delay))), key_count);

		auto t_before_swap = common::epoch_now();

		for ( unsigned i=0; i+1 < keys.size(); ++i )
		{
			auto value = common::random_string(16);
			auto key = common::random_string(8);
			PLOG("swapping keys (%s, %s)", keys[i].c_str(), keys[i+1].c_str());
			pool.swap_keys(keys[i], keys[i+1]);
			sleep(delay);
		}

		auto t_after_swap = common::epoch_now();
		t_after_swap.add_seconds(1); /* adjust for coarse granularity */

		PLOG(
			"Timestamps remaining from before swap [%lu .. %lu]"
			, t_before_swap.seconds()
			, t_after_swap.seconds()
		);
		key_count = 0;
		pool.map(
			[&key_count](
				const void* key
				, const size_t key_len
				, const void* // value
				, const size_t // value_len
				, const common::tsc_time_t timestamp
			) -> bool
			{
				++key_count;
				std::ostringstream ts;
				ts << timestamp;
				PLOG("Before swaps record: %.*s @ %s"
					, int(key_len), static_cast<const char *>(key), ts.str().c_str()
				);
				return true;
			}
			, 0
			, t_before_swap.sub_seconds(1) /* adjust for closed interval requirement */
		);
		EXPECT_EQ(cs->swap_updates_timestamp() ? 0 : keys.size(), key_count);

		key_count = 0;
		PLOG("Timestamps during swaps");
		pool.map(
			[&key_count](
				const void* key
				, const size_t key_len
				, const void* // value
				, const size_t // value_len
				, const common::tsc_time_t timestamp
			) -> bool
			{
				++key_count;
				std::ostringstream ts;
				ts << timestamp;
				PLOG("During swaps record: %.*s @ %s"
					, int(key_len), static_cast<const char *>(key), ts.str().c_str()
				);
				return true;
			}
			, t_before_swap
			, t_after_swap
		);
		EXPECT_EQ(cs->swap_updates_timestamp() ? keys.size() : 0, key_count);

		key_count = 0;
		PLOG("Timestamps after swaps");
		pool.map(
			[&key_count](
				const void* key
				, const size_t key_len
				, const void* // value
				, const size_t // value_len
				, const common::tsc_time_t timestamp
			) -> bool
			{
				++key_count;
				std::ostringstream ts;
				ts << timestamp;
				PLOG("After swaps record: %.*s @ %s"
					, int(key_len), static_cast<const char *>(key), ts.str().c_str()
				);
				return true;
			}
			, t_after_swap
			, 0
		);
		EXPECT_EQ(0, key_count);
	}

	PLOG("Closing pool.");
	EXPECT_EQ(S_OK, pool.close());
}

TEST_F(KVStore_test, Iterator)
{
	auto pool = create_pool(iterator_test_pool, MB(32));

	common::epoch_time_t now = 0;

	for(unsigned i=0;i<10;i++) {
	  auto value = common::random_string(16);
	  auto key = common::random_string(8);

	  if(i==5) { sleep(2); now = common::epoch_now(); }

	  FLOG("({}) adding key-value pair key({}) value({})", i, key, value);
	  pool.put(key, value.c_str(), value.size());
	}

	pool.map(
	              [](const void * key,
	                 const size_t key_len,
	                 const void * value,
	                 const size_t value_len) -> int
	              {
	                FINF("key:({} {}) value({})"
						, key
						, string_view(static_cast<const char *>(key), key_len)
						, string_view(static_cast<const char *>(value), value_len)
	                    , static_cast<const char *>(value));
	                return 0;
	              }
	              );

	PLOG("Iterating...");
	status_t rc;
	IKVStore::pool_reference_t ref;
	bool time_match;

	{
		auto iter = pool.open_iterator();
		while((rc = iter.deref(0, 0, ref, time_match, true)) == S_OK) {
			FLOG("iterator: key({}) value({}) {}"
				, string_view(static_cast<const char *>(ref.key), ref.key_len)
				, string_view(static_cast<const char *>(ref.value), ref.value_len)
				, ref.timestamp.seconds()
			);
		}
	}
	EXPECT_EQ(E_OUT_OF_BOUNDS, rc);

	{
		auto iter = pool.open_iterator();
		EXPECT_LT(0, now.seconds());
		while((rc = iter.deref(0, now, ref, time_match, true)) == S_OK) {
			FLOG("(time-constrained) iterator: key({}) value({}) {} (match={})"
				, string_view(static_cast<const char *>(ref.key), ref.key_len)
				, string_view(static_cast<const char *>(ref.value), ref.value_len)
				, ref.timestamp.seconds()
				, time_match ? "y":"n"
			);
		}

	}
	EXPECT_EQ(E_OUT_OF_BOUNDS, rc);

	PLOG("Disturbed iteration...");
	unsigned i=0;
	{
		auto iter = pool.open_iterator();
			while((rc = iter.deref(0, 0, ref, time_match, true)) == S_OK) {
				FLOG("iterator: key({}) value({}) {}"
					, string_view(static_cast<const char *>(ref.key), ref.key_len)
					, string_view(static_cast<const char *>(ref.value), ref.value_len)
					, ref.timestamp.seconds()
				);
				i++;
				if(i == 5) {
				/* disturb iteration */
				auto value = common::random_string(16);
				auto key = common::random_string(8);
				FLOG("adding key-value pair key({}) value({})", key, value);
				pool.put(key, value.c_str(), value.size());
			}
		}
		EXPECT_EQ(E_ITERATOR_DISTURBED, rc);
	}

	PLOG("Closing pool.");
	EXPECT_NE(IKVStore::POOL_ERROR, pool.handle());
	EXPECT_EQ(S_OK, pool.close());
}


TEST_F(KVStore_test, KeySwap)
{
	auto pool = create_pool("keyswap", MB(32));

	std::string left_key = "LeftKey";
	std::string right_key = "RightKey";
	std::string left_value = "This is left";
	std::string right_value = "This is right";

	ASSERT_OK(pool.put(left_key, left_value.c_str(), left_value.length()));
	ASSERT_OK(pool.put(right_key, right_value.c_str(), right_value.length()));

	ASSERT_OK(pool.swap_keys(left_key, right_key));

	iovec new_left{}, new_right{};
	pool.get(left_key, new_left.iov_base, new_left.iov_len);
	pool.get(right_key, new_right.iov_base, new_right.iov_len);

	FLOG("left: {}", string_view(static_cast<const char *>(new_left.iov_base), new_left.iov_len));
	FLOG("right: {}", string_view(static_cast<const char *>(new_right.iov_base), new_right.iov_len));
	EXPECT_EQ(0, strncmp(static_cast<const char *>(new_left.iov_base), right_value.c_str(), new_left.iov_len));
	EXPECT_EQ(0, strncmp(static_cast<const char *>(new_right.iov_base), left_value.c_str(), new_right.iov_len));
	pool.free_memory(new_right.iov_base);
	pool.free_memory(new_left.iov_base);

	EXPECT_EQ(S_OK, pool.close());
}

TEST_F(KVStore_test, AlignedLock)
{
	auto pool = create_pool_shared("alignedlock", MB(32));

	std::vector<size_t> alignments = {8,16,32,128,512,2048,4096};

	for(size_t alignment: alignments)
	{
		void * addr = nullptr;
		size_t value_len = 4096;

		{
			exclusive_lock l1(pool, "key", IKVStore::STORE_LOCK_WRITE, addr, value_len, alignment);
			EXPECT_EQ(S_OK_CREATED, l1.rc());
			EXPECT_EQ(true, check_aligned(addr, alignment));
			exclusive_lock l2(pool, "key", IKVStore::STORE_LOCK_WRITE, addr, value_len, alignment);
			EXPECT_EQ(E_LOCKED, l2.rc());
		}
		EXPECT_EQ(S_OK,pool->erase("key"));
	}

	EXPECT_EQ(S_OK, pool->close());
}

/*
 * Following tests were copied from test1 in hstore unit_test
 */

TEST_F(KVStore_test, CreatePool)
{
	auto pool_foo_name = "foo/" + pool_name();
	auto pool_foo = create_pool(pool_foo_name, 42, IKVStore::FLAGS_CREATE_ONLY, 42);
	{
		auto rc = pool_foo.close();
		EXPECT_EQ(S_OK, rc);
	}

	auto pool = create_pool(
		pool_name()
		, sv_count_to_size(single_value_size), IKVStore::FLAGS_CREATE_ONLY, estimated_object_count_small
	);
	auto pool2 = open_pool(pool_name());
	{
		void *v0;
		size_t s = 4096;
		auto r0 = pool.allocate_memory(8, s, v0);
		EXPECT_EQ(S_OK, r0);
		void *v1;
		auto r1 = pool.allocate_memory(8, s, v1);
		EXPECT_EQ(S_OK, r1);
		EXPECT_NE(v0, v1);
	}

	EXPECT_EQ(S_OK, pool2.close());
	EXPECT_EQ(S_OK, pool.close());

	/* All open handles closed */
	EXPECT_EQ(IKVStore::E_POOL_NOT_FOUND, pool2.close());

	auto pool_copy = open_pool(pool_name());
	ASSERT_LT(IKVStore::POOL_ERROR, pool_copy.handle());
	{
		std::list<std::string> nl;
		auto rc = _kvstore->get_pool_names(nl);
		std::cerr << "pool_names {";
		for ( auto n : nl )
		{
			std::cerr << n << ",";
		}
		std::cerr << "}\n";
		EXPECT_EQ(S_OK, rc);
	}
	{
		EXPECT_EQ(S_OK, _kvstore->delete_pool(pool_foo_name));
	}
}

TEST_F(KVStore_test, BasicGet0)
{
	auto pool = create_pool(
		pool_name()
		, sv_count_to_size(single_value_size), IKVStore::FLAGS_CREATE_ONLY, estimated_object_count_small
	);
	show_percent_used(pool);

	void * value = nullptr;
	size_t value_len = 0;

	EXPECT_EQ(IKVStore::E_KEY_NOT_FOUND, pool.get(single_key, value, value_len));
	pool.free_memory(value);
}

TEST_F(KVStore_test, BasicPutGetBig)
{
	auto pool = create_pool(
		pool_name()
		, sv_count_to_size(single_value_size)
		, IKVStore::FLAGS_CREATE_ONLY, estimated_object_count_small
	);

	EXPECT_EQ(S_OK, pool.put(single_key, single_value.data(), single_value.length()));

	void * value = nullptr;
	size_t value_len = 0;
	{
		auto r = pool.get(single_key, value, value_len);
		EXPECT_EQ(S_OK, r);

		if ( S_OK == r )
		{
			FINF("Value=({}) {:x}", string_view(static_cast<char *>(value), value_len), value_len);
			EXPECT_EQ(single_value.size(), value_len);
			EXPECT_EQ(0, memcmp(single_value.data(), value, single_value.size()));
			pool.free_memory(value);
		}
	}
}

TEST_F(KVStore_test, BasicPutLocked)
{
	/* put a value.
	 * read-lock it
	 * try to replace (expecte LOCKED)
	 * try to resize (expecte LOCKED)
	 * get (expect value is unchanged)
	 */
	auto pool = create_pool_shared(
		pool_name()
		, sv_count_to_size(single_value_size) // fixed
		, IKVStore::FLAGS_CREATE_ONLY
		, estimated_object_count_small
	);

	EXPECT_EQ(S_OK, pool->put(single_key, single_value.data(), single_value.length()));

	void *value0 = nullptr;
	std::size_t value0_len = 0;

	{
		size_t alignment = 0;
		shared_lock lk(pool, single_key, IKVStore::STORE_LOCK_READ, value0, value0_len, alignment);
		EXPECT_EQ(S_OK, lk.rc());
		if ( lk.rc() == S_OK )
		{
			EXPECT_EQ(0, memcmp(single_value.data(), value0, value0_len));
		}

		EXPECT_EQ(E_LOCKED, pool->put(single_key, single_value.data(), single_value.size()));

		EXPECT_EQ(cs->rc_resize_locked(), pool->resize_value(single_key, single_value.size(), 64)); /* mapstore: E_INVAL */

		{
			void * value = nullptr;
			size_t value_len = 0;
			auto r = pool->get(single_key, value, value_len);
			EXPECT_EQ(S_OK, r);
			if ( r == S_OK )
			{
				EXPECT_EQ(single_value.size(), value_len);
				EXPECT_EQ(0, memcmp(single_value.data(), value, value_len));
				pool->free_memory(value);
			}
		}

		EXPECT_EQ(S_OK, lk.unlock());
	}

	std::size_t small_size = 16; /* Small, but large enough to preserve all characters in "Hello world!" */

	/*
	 * resize value
	 * lock value
	 * get value (expect new size)
	 */

	/* resize to an inline size. */
	{
		auto r = pool->resize_value(single_key, small_size, 16);
		EXPECT_EQ(S_OK, r);
	}
	{
		void *v = nullptr;
		std::size_t v_len = 0;
		size_t alignment = 0;
		shared_lock lk(pool, single_key, IKVStore::STORE_LOCK_READ, v, v_len, alignment);
		EXPECT_EQ(S_OK, lk.rc());
		if ( lk.rc() == S_OK )
		{
			EXPECT_NE(nullptr, v);
			EXPECT_EQ(16, v_len);

			EXPECT_EQ(0, memcmp(single_value.data(), v, v_len));
#if 0
			/* alignment is just a suggestion, so we cannot insist on it, especiall for small-sized values */
			EXPECT_EQ(0, reinterpret_cast<std::size_t>(v) % 16);
#endif
			/* We cannot insist that v be unaligned to 256, but if it is aligned then the following test will verify nothing */
			EXPECT_NE(0, reinterpret_cast<std::size_t>(v) % 256);

			auto r = lk.unlock();
			EXPECT_EQ(S_OK, r);
		}
	}

	{
		void * value = nullptr;
		size_t value_len = 0;
		auto r = pool->get(single_key, value, value_len);
		EXPECT_EQ(S_OK, r);
		if ( r == S_OK )
		{
			EXPECT_EQ(small_size, value_len);
			EXPECT_EQ(0, memcmp(single_value.data(), value, value_len));
			pool->free_memory(value);
		}
	}

	/*
	 * resize value go single_value_size/2
	 * lock value
	 * get value (expect new size)
	 */
	{
		auto r = pool->resize_value(single_key, single_value.size() / 2, 256);
		EXPECT_EQ(S_OK, r);
	}
	{
		void *v = nullptr;
		std::size_t v_len = 0;
		size_t alignment = 0;
		shared_lock lk(pool, single_key, IKVStore::STORE_LOCK_READ, v, v_len, alignment);
		EXPECT_EQ(S_OK, lk.rc());
		if ( lk.rc() == S_OK )
		{
			EXPECT_NE(nullptr, v);
			EXPECT_EQ(single_value.size() / 2, v_len);
			EXPECT_EQ(0, memcmp(single_value.data(), v, v_len));
			EXPECT_EQ(0, reinterpret_cast<std::size_t>(v) % 256);
#if 0
			/* We cannot insist that v be unaligned to 1024, but if it is aligned then the following 1024 test will verify nothing */
			EXPECT_NE(0, reinterpret_cast<std::size_t>(v) % 1024);
#endif
			auto r = lk.unlock();
			EXPECT_EQ(S_OK, r);
		}
	}

	{
		void * value = nullptr;
		size_t value_len = 0;
		auto r = pool->get(single_key, value, value_len);
		EXPECT_EQ(S_OK, r);
		if ( r == S_OK )
		{
			EXPECT_EQ(single_value.size() / 2, value_len);
			EXPECT_EQ(0, memcmp(single_value.data(), value, value_len));
			pool->free_memory(value);
		}
	}

	/*
	 * resize value to single_value_size
	 * lock value
	 * get value (expect new size)
	 */
	{
		auto r = pool->resize_value(single_key, single_value.size(), 1024);
		EXPECT_EQ(S_OK, r);
	}
	{
		void *v = nullptr;
		std::size_t v_len = 0;
		size_t alignment = 0;
		shared_lock lk(pool, single_key, IKVStore::STORE_LOCK_READ, v, v_len, alignment);
		EXPECT_EQ(S_OK, lk.rc());
		if ( lk.rc() == S_OK )
		{
			EXPECT_NE(nullptr, v);
			EXPECT_EQ(single_value.size(), v_len);
			EXPECT_EQ(0, memcmp(single_value.data(), v, v_len));
			EXPECT_EQ(0, reinterpret_cast<std::size_t>(v) % 1024);
			auto r = lk.unlock();
			EXPECT_EQ(S_OK, r);
		}
	}

	{
		void * value = nullptr;
		size_t value_len = 0;
		auto r = pool->get(single_key, value, value_len);
		EXPECT_EQ(S_OK, r);
		if ( r == S_OK )
		{
			EXPECT_EQ(single_value.size(), value_len);
			EXPECT_EQ(0, memcmp(single_value.data(), value, single_value.size()));
			pool->free_memory(value);
		}
	}

	/* Reqeusted change in behavior: length 0 is now a magic value which means
	 * "do not try to create an element for the key, if not found"
	 */
	{
		void *v = nullptr;
		size_t value_len = 0;
		size_t alignment = 0;
		shared_lock lk(pool, missing_key, IKVStore::STORE_LOCK_READ, v, value_len, alignment);
		EXPECT_EQ(IKVStore::E_KEY_NOT_FOUND, lk.rc());
	}

}

TEST_F(KVStore_test, BasicGet2)
{
	auto pool = create_pool(
		pool_name()
		, sv_count_to_size(single_value_size )
		, IKVStore::FLAGS_CREATE_ONLY, estimated_object_count_small
	);

	EXPECT_EQ(S_OK, pool.put(single_key, single_value.data(), single_value.length()));

	void * value = nullptr;
	size_t value_len = 0;
	auto r = pool.get(single_key, value, value_len);
	EXPECT_EQ(S_OK, r);
	if ( S_OK == r )
	{
		FINF("Value=({}) {:x}", string_view(static_cast<char *>(value), value_len), value_len);
		EXPECT_EQ(single_value.size(), value_len);
		EXPECT_EQ(0, memcmp(single_value.data(), value, single_value.size()));
		pool.free_memory(value);
	}
}

/* hstore issue 41 specifies different implementations for same-size replace vs different-size replace. */
TEST_F(KVStore_test, BasicReplaceSameSize)
{
	auto pool = create_pool(
		pool_name()
		, sv_count_to_size(single_value_size)
		, IKVStore::FLAGS_CREATE_ONLY, estimated_object_count_small
	);

	EXPECT_EQ(S_OK, pool.put(single_key, single_value.data(), single_value.length()));

	{
		single_value_updated_same_size.resize(single_value_size);
		EXPECT_EQ(S_OK, pool.put(single_key, single_value_updated_same_size.data(), single_value_updated_same_size.length()));
	}
	void * value = nullptr;
	size_t value_len = 0;
	auto r = pool.get(single_key, value, value_len);
	EXPECT_EQ(S_OK, r);
	if ( S_OK == r )
	{
		PINF("Value=(%.*s) %zu", static_cast<int>(value_len), static_cast<char *>(value), value_len);
		EXPECT_EQ(0, memcmp(single_value_updated_same_size.data(), value, single_value_updated_same_size.size()));
		pool.free_memory(value);
	}
}

TEST_F(KVStore_test, BasicReplaceDifferentSize)
{
	auto pool = create_pool(
		pool_name()
		, sv_count_to_size(single_value_size), IKVStore::FLAGS_CREATE_ONLY, estimated_object_count_small
	);

	EXPECT_EQ(S_OK, pool.put(single_key, single_value.data(), single_value.length()));

	EXPECT_EQ(S_OK, pool.put(single_key, single_value_updated_different_size.data(), single_value_updated_different_size.length()));

	void * value = nullptr;
	size_t value_len = 0;
	auto r = pool.get(single_key, value, value_len);
	EXPECT_EQ(S_OK, r);
	if ( S_OK == r )
	{
		PINF("Value=(%.*s) %zu", static_cast<int>(value_len), static_cast<char *>(value), value_len);
		EXPECT_EQ(0, memcmp(single_value_updated_different_size.data(), value, single_value_updated_different_size.size()));
		pool.free_memory(value);
	}
}

std::size_t put_many(pool_instance &pool, const limited_size_view<std::vector<KVStore_test::kv_t>> &kvv)
{
	std::size_t many_count_actual = 0;

	for ( auto &kv : kvv )
	{
		const auto &key = std::get<0>(kv);
		const auto &value = std::get<1>(kv);
		void * old_value = nullptr;
		size_t old_value_len = 0;
		if ( S_OK == pool.get(key, old_value, old_value_len) )
		{
			pool.free_memory(old_value);
		}
		else
		{
			auto r = pool.put(key, value.data(), value.length());
			EXPECT_EQ(S_OK, r);
			if ( r == S_OK )
			{
				++many_count_actual;
			}
		}
	}
	return many_count_actual;
}

TEST_F(KVStore_test, PutMany)
{
	auto pool = create_pool_sized(
		pool_name()
		, sv_count_to_size(single_value.size()) /* fixed */
		, mc_count_to_size(many_count_target_proposed) /* variable */
		, IKVStore::FLAGS_CREATE_ONLY
		, estimated_object_count_small
	);
	ASSERT_LT(IKVStore::POOL_ERROR, pool.handle());
	const auto many_count_target = mc_size_to_count(pool.variable());
	const auto many_count_actual = put_many(pool, limited_size_view(kvv, many_count_target));
	EXPECT_LE(many_count_actual, many_count_target);
	EXPECT_LE(many_count_target * 99 / 100, many_count_actual);
}

TEST_F(KVStore_test, BasicMap2)
{
	auto pool = create_pool_sized(
		pool_name()
		, sv_count_to_size(single_value_size) /* fixed */
		, mc_count_to_size(many_count_target_proposed) /* variable */
		, IKVStore::FLAGS_CREATE_ONLY
		, estimated_object_count_small
	);
	ASSERT_LT(IKVStore::POOL_ERROR, pool.handle());
	const auto many_count_target = mc_size_to_count(pool.variable());
	const auto many_count_actual = put_many(pool, limited_size_view(kvv, many_count_target));

	EXPECT_EQ(S_OK, pool.put(single_key, single_value_updated_different_size.data(), single_value_updated_different_size.length()));

	auto value_len_sum = std::size_t();
	pool.map(
		[&value_len_sum] (
				const void * // key
				, const size_t // key_len
				, const void * // value
				, const size_t value_len
			) -> int
			{
				value_len_sum += value_len;
				return 0;
			}
		);
	EXPECT_EQ(single_value_updated_different_size.length() + many_count_actual * many_value_length, value_len_sum);
}

TEST_F(KVStore_test, BasicMapKeys)
{
	auto pool = create_pool_sized(
		pool_name()
		, sv_count_to_size(single_value_size) /* fixed */
		, mc_count_to_size(many_count_target_proposed) /* variable */
		, IKVStore::FLAGS_CREATE_ONLY
		, estimated_object_count_small
	);
	ASSERT_LT(IKVStore::POOL_ERROR, pool.handle());

	EXPECT_EQ(S_OK, pool.put(single_key, single_value.data(), single_value.length()));
	const auto many_count_target = mc_size_to_count(pool.variable());
	const auto many_count_actual = put_many(pool, limited_size_view(kvv, many_count_target));

	auto key_len_sum = std::size_t();
	pool.map_keys(
		[&key_len_sum](
				const std::string &key
		) -> int
		{
			key_len_sum += key.size();
			return 0;
		}
	);
	EXPECT_EQ(single_key.size() + many_count_actual * many_key_length, key_len_sum);
}

TEST_F(KVStore_test, Count1)
{
	auto pool = create_pool_sized(
		pool_name()
		, sv_count_to_size(single_value_size) /* fixed */
		, mc_count_to_size(many_count_target_proposed) /* variable */
		, IKVStore::FLAGS_CREATE_ONLY
		, estimated_object_count_small
	);
	ASSERT_LT(IKVStore::POOL_ERROR, pool.handle());
	const auto many_count_target = mc_size_to_count(pool.variable());
	const auto many_count_actual = put_many(pool, limited_size_view(kvv, many_count_target));

	/* count should reflect Put, PutMany */
	EXPECT_EQ(many_count_actual, pool.count());
}

TEST_F(KVStore_test, CountByBucket)
{
/* buckets implemented, but not available at the kvstore interface */
#if 0
	std::uint64_t count = 0;
	pool->debug(2 /* COUNT_BY_BUCKET */, reinterpret_cast<std::uint64_t>(&count));
	/* should reflect Put, PutMany */
	EXPECT_EQ(single_count + many_count_actual, count);
#endif
}

TEST_F(KVStore_test, ClosePool2)
{
	auto pool = create_pool_sized(
		pool_name()
		, sv_count_to_size(single_value_size) /* fixed */
		, mc_count_to_size(many_count_target_proposed) /* variable */
		, IKVStore::FLAGS_CREATE_ONLY
		, estimated_object_count_small
	);
	ASSERT_LT(IKVStore::POOL_ERROR, pool.handle());
	const auto many_count_target = mc_size_to_count(pool.variable());
	const auto many_count_actual = put_many(pool, limited_size_view(kvv, many_count_target));
	EXPECT_EQ(S_OK, pool.put(single_key, single_value_updated_different_size.data(), single_value_updated_different_size.length()));

	show_percent_used(pool);

	pool.close();

	auto pool2 = open_pool(pool_name(), 0);
	ASSERT_LT(IKVStore::POOL_ERROR, pool2.handle());

	show_percent_used(pool2);

	{
		auto count = pool2.count();
		/* count should reflect Put, PutMany */
		EXPECT_EQ(single_count + many_count_actual, count);
	}

	{
		void * value = nullptr;
		size_t value_len = 0;
		auto r = pool2.get(single_key, value, value_len);
		EXPECT_EQ(S_OK, r);
		if ( S_OK == r )
		{
			PINF("Value=(%.*s) %zu", static_cast<int>(value_len), static_cast<char *>(value), value_len);
			pool2.free_memory(value);
		}
	}
}

TEST_F(KVStore_test, BasicGetAttribute)
{
	auto pool = create_pool(
		pool_name()
		, sv_count_to_size(single_value_size)
		, IKVStore::FLAGS_CREATE_ONLY, estimated_object_count_small
	);

	EXPECT_EQ(S_OK, pool.put(single_key, single_value_updated_different_size.data(), single_value_updated_different_size.length()));

	{
		std::vector<uint64_t> attr;
		auto r = pool.get_attribute(IKVStore::VALUE_LEN, attr, &single_key);
		EXPECT_EQ(S_OK, r);
		if ( S_OK == r )
		{
			EXPECT_EQ(1, attr.size());
			if ( 1 == attr.size() )
			{
				EXPECT_EQ(attr[0], single_value_updated_different_size.length());
			}
		}
		r = pool.get_attribute(IKVStore::Attribute(0), attr, &single_key);
		EXPECT_EQ(cs->rc_unknown_attribute(), r);
		r = pool.get_attribute(IKVStore::VALUE_LEN, attr, nullptr);
		EXPECT_EQ(cs->rc_attribute_key_null_ptr(), r);
		r = pool.get_attribute(IKVStore::VALUE_LEN, attr, &missing_key);
		EXPECT_EQ(IKVStore::E_KEY_NOT_FOUND, r);
	}

	/* verify that item still exists */
	{ // 511
		void * value = nullptr;
		size_t value_len = 0;
		auto r = pool.get(single_key, value, value_len);
		EXPECT_EQ(S_OK, r);
		if ( S_OK == r )
		{
			PINF("Value=(%.*s) %zu", static_cast<int>(value_len), static_cast<char *>(value), value_len);
			pool.free_memory(value);
		}
	}
}

TEST_F(KVStore_test, ResizeAttribute)
{
	auto pool = create_pool(
		pool_name()
		, sv_count_to_size(single_value_size)
		, IKVStore::FLAGS_CREATE_ONLY, estimated_object_count_small
	);

	EXPECT_EQ(S_OK, pool.put(single_key, single_value_updated_different_size.data(), single_value_updated_different_size.length()));

	{
		std::vector<uint64_t> attr0;
		auto r0 = pool.get_attribute(IKVStore::AUTO_HASHTABLE_EXPANSION, attr0, nullptr);
		EXPECT_EQ(cs->rc_attribute_hashtable_expansion(), r0);
		if ( r0 == S_OK )
		{
			ASSERT_EQ(1, attr0.size());
			EXPECT_EQ(1, attr0[0]);

			{
				std::vector<uint64_t> attr{0};
				EXPECT_EQ(S_OK, pool.set_attribute(IKVStore::AUTO_HASHTABLE_EXPANSION, attr, nullptr));
				EXPECT_EQ(1, attr.size());
			}

			{
				std::vector<uint64_t> attr;
				EXPECT_EQ(S_OK, pool.get_attribute(IKVStore::AUTO_HASHTABLE_EXPANSION, attr, nullptr));
				EXPECT_EQ(1, attr.size());
				if ( 1 == attr.size() )
				{
					EXPECT_EQ(0, attr[0]);
				}
			}

			{
				std::vector<uint64_t> attr{34};
				EXPECT_EQ(S_OK, pool.set_attribute(IKVStore::AUTO_HASHTABLE_EXPANSION, attr, nullptr));
				EXPECT_EQ(1, attr.size());
			}

			{
				std::vector<uint64_t> attr;
				EXPECT_EQ(S_OK, pool.get_attribute(IKVStore::AUTO_HASHTABLE_EXPANSION, attr, nullptr));
				EXPECT_EQ(1, attr.size());
				if ( 1 == attr.size() )
				{					
					EXPECT_EQ(1, attr[0]);
				}
			}
		}
	}
}

TEST_F(KVStore_test, GetMany)
{
	auto pool = create_pool_sized(
		pool_name()
		, sv_count_to_size(single_value_size) /* fixed */
		, mc_count_to_size(many_count_target_proposed) /* variable */
		, IKVStore::FLAGS_CREATE_ONLY
		, estimated_object_count_small
	);
	ASSERT_LT(IKVStore::POOL_ERROR, pool.handle());

	EXPECT_EQ(S_OK, pool.put(single_key, single_value_updated_different_size.data(), single_value_updated_different_size.length()));

	const auto many_count_target = mc_size_to_count(pool.variable());
	const limited_size_view lsv(kvv, many_count_target);
	const auto many_count_actual = put_many(pool, lsv);

	auto count = pool.count();
	/* count should reflect PutMany */
	EXPECT_EQ(single_count + many_count_actual, count);

	show_percent_used(pool);

	for ( auto i = 0; i != get_expand; ++i )
	{
		std::size_t mismatch_count = 0;
		for ( auto &kv : lsv )
		{
			const auto &key = std::get<0>(kv);
			const auto &ev = std::get<1>(kv);
			char value[many_value_length * 2];
			std::size_t value_len = many_value_length * 2;
			void *vp = value;
			auto r = pool.get(key, vp, value_len);
			EXPECT_EQ(S_OK, r);

			if ( r != S_OK ) break;

			EXPECT_EQ(vp, static_cast<void *>(value));
			EXPECT_EQ(ev.size(), value_len);
			mismatch_count += ( ev.size() != value_len || 0 != memcmp(ev.data(), value, ev.size()) );
		}
		EXPECT_EQ(lsv.size(), many_count_actual + mismatch_count);
	}
}

TEST_F(KVStore_test, GetManyPreallocated)
{
	auto pool = create_pool_sized(
		pool_name()
		, sv_count_to_size(single_value_size) /* fixed */
		, mc_count_to_size(many_count_target_proposed) /* variable */
		, IKVStore::FLAGS_CREATE_ONLY
		, many_count_target_proposed
	);
	ASSERT_LT(IKVStore::POOL_ERROR, pool.handle());

	EXPECT_EQ(S_OK, pool.put(single_key, single_value_updated_different_size.data(), single_value_updated_different_size.length()));

	const auto many_count_target = mc_size_to_count(pool.variable());
	const limited_size_view lsv(kvv, many_count_target);
	const auto many_count_actual = put_many(pool, lsv);

	auto count = pool.count();
	/* count should reflect PutMany */
	EXPECT_EQ(single_count + many_count_actual, count);

	show_percent_used(pool);

	for ( auto i = 0; i != get_expand; ++i )
	{
		std::size_t mismatch_count = 0;
		for ( auto &kv : lsv )
		{
			const auto &key = std::get<0>(kv);
			const auto &ev = std::get<1>(kv);
			char value[many_value_length * 2];
			std::size_t value_len = many_value_length * 2;
			void *vp = value;
			auto r = pool.get(key, vp, value_len);
			EXPECT_EQ(S_OK, r);

			if ( r != S_OK ) break;

			EXPECT_EQ(vp, static_cast<void *>(value));
			EXPECT_EQ(ev.size(), value_len);
			mismatch_count += ( ev.size() != value_len || 0 != memcmp(ev.data(), value, ev.size()) );
		}
		EXPECT_EQ(lsv.size(), many_count_actual + mismatch_count);
	}
}

TEST_F(KVStore_test, GetManyAllocating)
{
	auto pool = create_pool_sized(
		pool_name()
		, sv_count_to_size(single_value_size) /* fixed */
		, mc_count_to_size(many_count_target_proposed) /* variable */
		, IKVStore::FLAGS_CREATE_ONLY
		, estimated_object_count_small
	);
	ASSERT_LT(IKVStore::POOL_ERROR, pool.handle());

	EXPECT_EQ(S_OK, pool.put(single_key, single_value_updated_different_size.data(), single_value_updated_different_size.length()));

	const auto many_count_target = mc_size_to_count(pool.variable());
	const limited_size_view lsv(kvv, many_count_target);
	const auto many_count_actual = put_many(pool, lsv);

	for ( auto i = 0; i != get_expand; ++i )
	{
		std::size_t mismatch_count = 0;
		for ( auto &kv : lsv )
		{
			const auto &key = std::get<0>(kv);
			const auto &ev = std::get<1>(kv);
			void * value = nullptr;
			std::size_t value_len = 0;
			auto r = pool.get(key, value, value_len);
			EXPECT_EQ(S_OK, r);

			if ( r != S_OK ) break;

			EXPECT_EQ(ev.size(), value_len);
			mismatch_count += ( ev.size() != value_len || 0 != memcmp(ev.data(), value, ev.size()) );
			pool.free_memory(value);
		}
		EXPECT_EQ(lsv.size(), many_count_actual + mismatch_count);
	}
}

TEST_F(KVStore_test, GetDirectMany)
{
	auto pool = create_pool_sized(
		pool_name()
		, sv_count_to_size(single_value_size) /* fixed */
		, mc_count_to_size(many_count_target_proposed) /* variable */
		, IKVStore::FLAGS_CREATE_ONLY
		, estimated_object_count_small
	);
	ASSERT_LT(IKVStore::POOL_ERROR, pool.handle());

	EXPECT_EQ(S_OK, pool.put(single_key, single_value_updated_different_size.data(), single_value_updated_different_size.length()));

	const auto many_count_target = mc_size_to_count(pool.variable());
	const limited_size_view lsv(kvv, many_count_target);
	const auto many_count_actual = put_many(pool, lsv);

	for ( auto i = 0; i != get_expand; ++i )
	{
		std::size_t mismatch_count = 0;
		for ( auto &kv : lsv )
		{
			const auto &key = std::get<0>(kv);
			const auto &ev = std::get<1>(kv);
			char value[many_value_length * 2];
			size_t value_len = many_value_length * 2;
			auto r = pool.get_direct(key, value, value_len);
			EXPECT_EQ(S_OK, r);

			if ( r != S_OK ) break;

			EXPECT_EQ(ev.size(), value_len);
			mismatch_count += ( ev.size() != value_len || 0 != memcmp(ev.data(), value, ev.size()) );
		}
		EXPECT_EQ(lsv.size(), many_count_actual + mismatch_count);
	}
}

TEST_F(KVStore_test, GetRegions)
{
	auto pool = create_pool_sized(
		pool_name()
		, sv_count_to_size(single_value_size) /* fixed */
		, mc_count_to_size(many_count_target_proposed) /* variable */
		, IKVStore::FLAGS_CREATE_ONLY
		, estimated_object_count_small
	);

	const auto many_count_target = mc_size_to_count(pool.variable());

	nupm::region_descriptor v;
	auto r = pool.get_regions(v);
	EXPECT_EQ(S_OK, r);
	if ( S_OK == r )
	{
		EXPECT_EQ(1, v.address_map().size());
		if ( 1 == v.address_map().size() )
		{
			PMAJOR("Pool region at %p len %zu", ::base(v.address_map().front()), ::size(v.address_map().front()));
			auto iov_base = reinterpret_cast<std::uintptr_t>(::base(v.address_map().front()));
			/* region no longer needs to be well-aligned, but heap_cc still aligns to a
			 * page boundary.
			 */
			EXPECT_EQ(iov_base & 0xfff, 0);
			EXPECT_GT(::size(v.address_map().front()), mc_count_to_size(many_count_target)/8U);
			EXPECT_LT(::size(v.address_map().front()), GiB(512));
		}
	}
}

TEST_F(KVStore_test, LockMany)
{
	auto pool = create_pool_sized_shared(
		pool_name()
		, sv_count_to_size(single_value_size) /* fixed */
		, mc_count_to_size(many_count_target_proposed) /* variable */
		, IKVStore::FLAGS_CREATE_ONLY
		, estimated_object_count_small
	);
	ASSERT_LT(IKVStore::POOL_ERROR, pool->handle());

	const auto many_count_target = mc_size_to_count(pool->variable());
	const auto many_count_actual = put_many(*pool, limited_size_view(kvv, many_count_target));

	const std::size_t lock_count = std::min(many_count_actual, std::size_t(60));

	/* Lock for read (should succeed)
	 * Lock again for read (should succeed).
	 * Lock for write (should fail).
	 * Lock a non-existent key for write, creating the key).
	 *
	 * Undo the three successful locks.
	 */
	unsigned ct = 0;
	for ( auto &kv : kvv )
	{
		if ( ct == lock_count ) { break; }
		const auto &key = std::get<0>(kv);
		const auto &ev = std::get<1>(kv);
		const auto key_new = std::get<0>(kv) + "x";
		void *value0 = nullptr;
		std::size_t value0_len = 0;

		size_t alignment = 0;
		shared_lock r0(pool, key, IKVStore::STORE_LOCK_READ, value0, value0_len, alignment);
		EXPECT_EQ(S_OK, r0.rc());
		if ( S_OK == r0.rc() )
		{
			EXPECT_EQ(many_value_length, value0_len);
			EXPECT_EQ(0, memcmp(ev.data(), value0, ev.size()));
			EXPECT_EQ(0, memcmp(key.data(), r0.key(), key.size()));
		}
		void * value1 = nullptr;
		std::size_t value1_len = 0;
		alignment = 0;
		shared_lock r1(pool, key, IKVStore::STORE_LOCK_READ, value1, value1_len, alignment);
		EXPECT_EQ(S_OK, r1.rc());
		if ( S_OK == r1.rc() )
		{
			EXPECT_EQ(many_value_length, value1_len);
			EXPECT_EQ(0, memcmp(ev.data(), value1, ev.size()));
			EXPECT_EQ(0, memcmp(key.data(), r1.key(), key.size()));
		}
		/* Exclusive locking test. */

		void * value2 = nullptr;
		std::size_t value2_len = 0;
		alignment = 0;
		exclusive_lock r2(pool, key, IKVStore::STORE_LOCK_WRITE, value2, value2_len, alignment);
		/* Undocumented behavior: lock conflict returns E_LOCKED */
		EXPECT_EQ(E_LOCKED, r2.rc());

		void * value3 = nullptr;
		std::size_t value3_len = many_value_length;
		try
		{
			alignment = 0;
			exclusive_lock r3(pool, key_new, IKVStore::STORE_LOCK_WRITE, value3, value3_len, alignment);
			/* Used to return S_OK; no longer does so */
			EXPECT_EQ(S_OK_CREATED, r3.rc());
			if ( S_OK == r3.rc() )
			{
				EXPECT_EQ(many_value_length, value3_len);
				EXPECT_NE(nullptr, value3);
				EXPECT_EQ(0, memcmp(key_new.data(), r3.key(), key_new.size()));
			}
			++ct;
		}
		catch (const std::system_error &)
		{
			/* Lock a non-existent key will fail if the key locked for read needs to be moved. */
		}
	}
}

TEST_F(KVStore_test, Size2c)
{
	auto pool = create_pool_sized(
		pool_name()
		, sv_count_to_size(single_value_size) /* fixed */
		, mc_count_to_size(many_count_target_proposed) /* variable */
		, IKVStore::FLAGS_CREATE_ONLY
		, estimated_object_count_small
	);
	ASSERT_LT(IKVStore::POOL_ERROR, pool.handle());

	EXPECT_EQ(S_OK, pool.put(single_key, single_value_updated_different_size.data(), single_value_updated_different_size.length()));

	const auto many_count_target = mc_size_to_count(pool.variable());
	const auto many_count_actual = put_many(pool, limited_size_view(kvv, many_count_target));

	/* verify that item still exists */
	{ // 770
		void * value = nullptr;
		size_t value_len = 0;
		auto r = pool.get(single_key, value, value_len);
		EXPECT_EQ(S_OK, r);
		if ( S_OK == r )
		{
			PINF("line %d Value=(%.*s) %zu", __LINE__, static_cast<int>(value_len), static_cast<char *>(value), value_len);
			pool.free_memory(value);
		}
	}
	auto count = pool.count();
	/* count should reflect put(single_key ...), put_many */
	EXPECT_EQ(single_count + many_count_actual, count);
}

/* Missing:
 *  - test of invalid parameters
 *    - offsets greater than sizeof data
 *    - non-existent key
 *    - invalid operations
 *  - test of crash recovery
 */
TEST_F(KVStore_test, BasicUpdate)
{
	auto pool = create_pool_sized(
		pool_name()
		, sv_count_to_size(single_value_size) /* fixed */
		, mc_count_to_size(many_count_target_proposed) /* variable */
		, IKVStore::FLAGS_CREATE_ONLY
		, estimated_object_count_small
	);
	ASSERT_LT(IKVStore::POOL_ERROR, pool.handle());

	EXPECT_EQ(S_OK, pool.put(single_key, single_value_updated_different_size.data(), single_value_updated_different_size.length()));

	const auto many_count_target = mc_size_to_count(pool.variable());
	const auto many_count_actual = put_many(pool, limited_size_view(kvv, many_count_target));

	/* verify that item still exists */
	{ // 796
		void * value = nullptr;
		size_t value_len = 0;
		auto r = pool.get(single_key, value, value_len);
		EXPECT_EQ(S_OK, r);
		if ( S_OK == r )
		{
			PINF("line %d Value=(%.*s) %zu", __LINE__, static_cast<int>(value_len), static_cast<char *>(value), value_len);
			pool.free_memory(value);
		}
	}
	{
		std::vector<std::unique_ptr<IKVStore::Operation>> v;
		v.emplace_back(std::make_unique<IKVStore::Operation_write>(0, 1, "W"));
		v.emplace_back(std::make_unique<IKVStore::Operation_write>(2, 3, "XYZ"));
		std::vector<IKVStore::Operation *> v2;
		std::transform(v.begin(), v.end(), std::back_inserter(v2), [] (const auto &i) { return i.get(); });
		auto r0 =
			pool.atomic_update(
				single_key
				, v2
			);
		EXPECT_EQ(cs->rc_atomic_update(), r0);
		if ( r0 == S_OK )
		{
			void * value = nullptr;
			size_t value_len = 0;
			auto r = pool.get(single_key, value, value_len);
			EXPECT_EQ(S_OK, r);
			if ( S_OK == r )
			{
				PINF("Value=(%.*s) %zu", static_cast<int>(value_len), static_cast<char *>(value), value_len);
				EXPECT_EQ(single_value_updated_different_size.size(), value_len);
				EXPECT_EQ(0, memcmp(single_value_updated3.data(), value, single_value_updated3.size()));
				pool.free_memory(value);
			}
		}
	}

	auto count = pool.count();
	EXPECT_EQ(single_count + many_count_actual, count);
}

TEST_F(KVStore_test, BasicErase)
{
	auto pool = create_pool_sized(
		pool_name()
		, sv_count_to_size(single_value_size) /* fixed */
		, mc_count_to_size(many_count_target_proposed) /* variable */
		, IKVStore::FLAGS_CREATE_ONLY
		, estimated_object_count_small
	);
	ASSERT_LT(IKVStore::POOL_ERROR, pool.handle());

	EXPECT_EQ(S_OK, pool.put(single_key, single_value_updated_different_size.data(), single_value_updated_different_size.length()));

	const auto many_count_target = mc_size_to_count(pool.variable());
	const auto many_count_actual = put_many(pool, limited_size_view(kvv, many_count_target));

	show_percent_used(pool);

	EXPECT_EQ(S_OK, pool.erase(single_key));

	EXPECT_EQ(many_count_actual, pool.count());
}

TEST_F(KVStore_test, EraseMany)
{
	auto pool = create_pool_sized(
		pool_name()
		, sv_count_to_size(single_value_size) /* fixed */
		, mc_count_to_size(many_count_target_proposed) /* variable */
		, IKVStore::FLAGS_CREATE_ONLY
		, estimated_object_count_small
	);
	ASSERT_LT(IKVStore::POOL_ERROR, pool.handle());

	const auto many_count_target = mc_size_to_count(pool.variable());
	const auto many_count_actual = put_many(pool, limited_size_view(kvv, many_count_target));

	auto erase_count = 0;
	for ( auto &kv : kvv )
	{
		const auto &key = std::get<0>(kv);
		auto r = pool.erase(key);
		if ( r == S_OK )
		{
			++erase_count;
		}
	}
	EXPECT_LE(many_count_actual, erase_count);
	EXPECT_EQ(0, pool.count());
}

TEST_F(KVStore_test, AllocDealloc4K)
{
	auto pool = create_pool_sized(
		pool_name()
		, sv_count_to_size(single_value_size) /* fixed */
		, mc_count_to_size(many_count_target_proposed) /* variable */
		, IKVStore::FLAGS_CREATE_ONLY
		, estimated_object_count_small
	);
	ASSERT_LT(IKVStore::POOL_ERROR, pool.handle());

	show_percent_used(pool);

	std::set<void *> allocations{};

	/* Many 4K-aligned allocations */
	for ( auto i = 0; i != 1000; ++i )
	{
		void *v = nullptr;
		size_t s = 4096;
		auto r = pool.allocate_memory(8, s, v);
		EXPECT_EQ(S_OK, r);
		EXPECT_EQ(0, allocations.count(v));
		allocations.insert(v);
		EXPECT_EQ(1, allocations.count(v));
	}

	/* close and reopen the pool */
	{
		auto rc = pool.close();
		EXPECT_EQ(S_OK, rc);

		auto pool2 = open_pool(pool_name());
		ASSERT_LT(0, int64_t(pool2.handle()));

		/* Many 4K-aligned frees */
		for ( auto v : allocations )
		{
			auto r = pool2.free_memory(v, 8);
			EXPECT_EQ(S_OK, r);
		}
	}
}

TEST_F(KVStore_test, AllocDealloc)
{
	auto pool = create_pool_sized(
		pool_name()
		, sv_count_to_size(single_value_size) /* fixed */
		, mc_count_to_size(many_count_target_proposed) /* variable */
		, IKVStore::FLAGS_CREATE_ONLY
		, estimated_object_count_small
	);
	ASSERT_LT(IKVStore::POOL_ERROR, pool.handle());

	show_percent_used(pool);

	status_t r = S_OK;
#if 0
	/* alignment is now a "hint", so 0 alignment is no longer an error */
	void *v = nullptr;
	/* In this function alignment is now a "hint", so 0 alignment is no longer an error */
	r = pool->allocate_memory(100, 0, v);
	/* exceedingly small alignment is also no longer an error */
	auto r = pool->allocate_memory(100, 1, v);
	EXPECT_EQ(IKVStore::E_BAD_ALIGNMENT, r); /* alignment less than sizeof(void *) */
#endif

	/* allocate various sizes */
	void *v128 = nullptr;
	void *v160 = nullptr;
	void *v704 = nullptr;
	void *v2048 = nullptr;
	void *v0 = nullptr;
	r = pool.allocate_memory(128, 32, v128);
	EXPECT_EQ(S_OK, r);
	EXPECT_NE(v128, nullptr);
	r = pool.allocate_memory(160, 8, v160);
	EXPECT_EQ(S_OK, r);
	EXPECT_NE(v160, nullptr);
	r = pool.allocate_memory(704, 64, v704);
	EXPECT_EQ(S_OK, r);
	EXPECT_NE(v704, nullptr);
	r = pool.allocate_memory(2048, 2048, v2048);
	EXPECT_EQ(S_OK, r);
	EXPECT_NE(v2048, nullptr);
	auto r0 = pool.allocate_memory(0, 8, v0);
	EXPECT_EQ(cs->rc_allocate_pool_memory_size_0(), r0); /* mapstore fails */
	if ( r0 == S_OK )
	{
		EXPECT_NE(v0, nullptr);
	}
	else
	{
		EXPECT_EQ(v0, nullptr);
	}

	/* close and reopen the pool */
	{
		auto rc = pool.close();
		EXPECT_EQ(S_OK, rc);

		auto pool2 = open_pool(pool_name());
		ASSERT_LT(IKVStore::POOL_ERROR, pool2.handle());

		if ( r0 == S_OK && v0 ) { EXPECT_EQ(S_OK, pool2.free_memory(v0, 0)); }
		if ( v2048 ) { EXPECT_EQ(S_OK, pool2.free_memory(v2048, 2048)); }
		if ( v704 ) { EXPECT_EQ(S_OK, pool2.free_memory(v704, 704)); }
		if ( v160 ) { EXPECT_EQ(S_OK, pool2.free_memory(v160, 160)); }
		if ( v128 ) { EXPECT_EQ(S_OK, pool2.free_memory(v128, 128)); }
#if 0
		/* Removed: unless the free logic validates arguments, might
		 * corrupt memory rather than failing.
		 */
		if ( v128 ) { EXPECT_EQ(E_INVALID_ARG, pool2.free_memory(v128, 128)); } /* not found */
#endif
	}
}

TEST_F(KVStore_test, DeletePool)
{
	auto pool = create_pool(
		pool_name()
		, sv_count_to_size(single_value_size)
		, IKVStore::FLAGS_CREATE_ONLY, estimated_object_count_small
	);

	show_percent_used(pool);

	pool.close();
	_kvstore->delete_pool(pool_name());
}

TEST_F(KVStore_test, OutOfMemory)
{
	auto pool = create_pool("oom", MiB(32), IKVStore::FLAGS_CREATE_ONLY);

	{
		status_t rc = S_OK;
		while ( rc == S_OK )
		{
			void *p = 0;
			rc = pool.allocate_memory(MiB(1), 8, p);
			(void) p;
		}
		EXPECT_EQ(cs->rc_out_of_memory(), rc); /* IMVStore::E_TOO_LARGE or E_INVAL */
	}
	ASSERT_EQ(S_OK, pool.close());
}

TEST_F(KVStore_test, NumaMask)
{
	auto pool = create_pool("numaMask", MiB(1));

	auto it = mc.find(+IKVStore_factory::k_numa_nodes);
	if ( it != mc.end() && it->second.size() != 0 )
	{
		std::vector<uint64_t> numa_mask_result;
		auto r = pool.get_attribute(component::IKVStore::NUMA_MASK, numa_mask_result);
		if ( r == S_OK )
		{
			EXPECT_EQ(1, numa_mask_result.size());
			if ( 1 == numa_mask_result.size() )
			{
				auto numa_mask = numa_mask_result[0];
				/* a trivial mask (1 node, less than 3 chars) should yield exactly one 1 bit */
				if ( it->second.size() < 3 )
		        {
					EXPECT_NE(0, numa_mask);
					EXPECT_EQ(0, (numa_mask & (numa_mask-1)));
				}
				std::cout << "Numa node mask is " << std::hex << numa_mask << "\n";
			}
		}
	}
}

} // namespace

namespace c_json = common::json;
using json = c_json::serializer<c_json::dummy_writer>;
json::array devdax_location
	( json::object
		( json::member("path", "/dev/dax0.0")
		, json::member("addr", 0x9000000000)
		)
	);

auto fsdax_location =
	json::array
	( json::object
		( json::member("path", "/mnt/pmem1")
		, json::member("addr", 0x9000000000)
		)
	);

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);

	namespace po = boost::program_options;

	po::options_description            desc("Options");
	po::positional_options_description g_pos; /* no positional options */

	desc.add_options()
		("help,h", "Show help")
		("numa-nodes", po::value<std::string>(), "Numa node specification (mapstore only)")
		("store", po::value<std::string>()->default_value("mapstore"), "Store type to test: e.g., hstore, hstore-cc, mapstore")
		("daxconfig"
			, po::value<std::string>()->default_value(
				json::array(
					json::object(
						json::member("path", "/dev/dax0.0")
						, json::member("addr", 0x9000000000)
					)
		        ).str()
			)
			, "dax configuration (hstore* only), in JSON. default [{\"path\": \"/dev/dax0.0\", \"addr\": \"0x9000000000\"}]"
		)
		("mm-plugin-path", po::value<std::string>(), "path to memory manager plugin")
		("debug", po::value<std::string>(), "store debug level")
		("many-count", po::value<std::size_t>()->default_value(2000000), "number of KV pairs to test for \"small KV pair, large quantitiy\" tests")
		;

	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(desc).positional(g_pos).run(), vm);

	if (vm.count("help") > 0)
	{
	    std::cout << desc;
	  	return -1;
	}

	store = vm["store"].as<std::string>();
	using IKVStore_factory = component::IKVStore_factory;
	mc = IKVStore_factory::map_create
		{
			{+IKVStore_factory::k_name, "numa0"},
			{+IKVStore_factory::k_dax_config, vm["daxconfig"].as<std::string>()}
		};
	if ( vm.count("numa-nodes") )
	{
		mc.insert( {+IKVStore_factory::k_numa_nodes, vm["numa-nodes"].as<std::string>()} );
	}
	if ( vm.count("mm-plugin-path") )
	{
        mc.insert( {+IKVStore_factory::k_mm_plugin_path, vm["mm-plugin-path"].as<std::string>()} );
	}
	if ( vm.count("debug") )
	{
        mc.insert( {+IKVStore_factory::k_debug, vm["debug"].as<std::string>()} );
	}
	many_count_proposed = vm["many-count"].as<std::size_t>();

	cs = locate_custom_store(store);
	auto r = RUN_ALL_TESTS();

	return r;
}
