/* note: we do not include component source, only the API definition */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <ccpm/cca.h>
#include <common/env.h> /* env_value */
#include <common/errors.h>
#include <common/logging.h>
#include <common/utils.h>

//#define GPERF_TOOLS

#ifdef GPERF_TOOLS
#include <gperftools/profiler.h>
#endif
#include <libpmem.h>
#include <cstdlib> // alligned_alloc
#include <iostream> // cerr
#include <string> // stoull

struct {
  uint64_t uuid;
} Options;

namespace
{
	struct persister_test2 final
		: public ccpm::persister
	{
		void persist(common::byte_span s) override
		{
			::pmem_persist(::base(s), ::size(s));
		}
	};

	persister_test2 p2{};
}

// The fixture for testing class Foo.
class Libccpm_test : public ::testing::Test {
 protected:
  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  virtual void SetUp()
  {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  virtual void TearDown()
  {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }

  // Objects declared here can be used by all tests in the test case
};


TEST_F(Libccpm_test, ccpm_cca_scenario_A)
{
	std::size_t size_mb = common::env_value<unsigned>("SIZE_MB",128);
	/*
	 * Values of SIZE_MB larger than available DRAM and/or swap space 
	 * may be accomodated by changing value of /proc/sys/vm/overcommit_memory,
	 * e.g.
	 *   # echo 1 > /proc/sys/vm/overcommit_memory
	 * to remove all limits on overcommit.
	 */
	std::cerr << "size is " << size_mb << "MiB" << "\n";
  std::size_t size = MiB(size_mb);
  auto pr = aligned_alloc(4096,size);
  ASSERT_NE(nullptr, pr);

  ccpm::region_vector_t rv(
    ccpm::region_vector_t::value_type(
      common::make_byte_span(pr, size)
    )
  );
  {
    ccpm::cca ccheap(&p2, rv);

    void  * p = nullptr;

    EXPECT_EQ(S_OK, ccheap.allocate(p,1024,8));
    EXPECT_NE(nullptr, p);
    p = nullptr;

    EXPECT_EQ(S_OK, ccheap.allocate(p,1024,8));
    EXPECT_NE(nullptr, p);
    p = nullptr;

    EXPECT_EQ(S_OK, ccheap.allocate(p,328,8));
    EXPECT_NE(nullptr, p);
    p = nullptr;

    EXPECT_EQ(S_OK, ccheap.allocate(p,1024,8));
    EXPECT_NE(nullptr, p);
    p = nullptr;

    EXPECT_EQ(S_OK, ccheap.allocate(p,472,8));
    EXPECT_NE(nullptr, p);
    p = nullptr;

    PLOG("allocations OK");

		std::vector<std::pair<int,std::vector<void *>>> alloc_sizes;
		for ( int exp = 30; exp >= 0; exp -= 4 )
		{
			alloc_sizes.push_back({exp,std::vector<void *>{}});
			auto &allocs = alloc_sizes.back().second;
			auto rc = ccheap.allocate(p, 2UL<<exp, 8);
			while ( rc == S_OK )
			{
				allocs.push_back(p);
				p = nullptr;
				rc = ccheap.allocate(p, 2UL<<exp, 8);
			}

			ccheap.print(std::cerr);
			PLOG("allocation: 2^%u: %zu", exp, allocs.size());
		}

		for ( ; ! alloc_sizes.empty() ; alloc_sizes.pop_back() )
		{
			auto exp = alloc_sizes.back().first;
			auto &allocs = alloc_sizes.back().second;
			for ( ; ! allocs.empty(); allocs.pop_back() )
			{
				ccheap.free(allocs.back(), 2UL<<exp);
				EXPECT_EQ(nullptr, allocs.back());
			}
			
		}
  }
}

TEST_F(Libccpm_test, ccpm_cca)
{
  std::size_t size = 409600;
  auto pr = aligned_alloc(4096,size);
  ASSERT_NE(nullptr, pr);
  const void *ph = static_cast<const char *>(pr) + size;
  ccpm::region_vector_t rv(
    ccpm::region_vector_t::value_type(
      common::make_byte_span(pr, size)
    )
  );
  {
  int r = S_OK;
  ccpm::cca bt(&p2, rv);
  auto remain_cb = [&] () {
      std::size_t remain;
      r = bt.remaining(remain);
      PLOG("Remaining : %zu of %zu", remain, size);
      EXPECT_EQ(S_OK, r);
      EXPECT_LT(0, remain);
      EXPECT_GT(size, remain);
      return remain;
    };
  std::size_t remain0 = remain_cb();
  bt.print(std::cerr);

  void *p8 = nullptr;
  r = bt.allocate(p8, 8, 8);
  EXPECT_EQ(S_OK, r);
  EXPECT_NE(nullptr, p8);
  EXPECT_LE(pr, p8);
  EXPECT_GT(ph, p8);
  std::size_t remain1 = remain_cb();
  EXPECT_GE(remain0, remain1);
  bt.print(std::cerr);

  void *p16 = nullptr;
  r = bt.allocate(p16, 16, 16);
  EXPECT_EQ(S_OK, r);
  EXPECT_NE(nullptr, p16);
  EXPECT_NE(p8, p16);
  EXPECT_NE(nullptr, p16);
  EXPECT_LE(pr, p16);
  std::size_t remain2 = remain_cb();
  EXPECT_GE(remain1, remain2);
  bt.print(std::cerr);

  r = bt.free(p8, 8);
  EXPECT_EQ(S_OK, r);
  EXPECT_EQ(nullptr, p8);
  std::size_t remain3 = remain_cb();
  EXPECT_LE(remain2, remain3);

  r = bt.free(p16, 16);
  EXPECT_EQ(S_OK, r);
  EXPECT_EQ(nullptr, p16);
  std::size_t remain4 = remain_cb();
  EXPECT_LE(remain3, remain4);
  }
  {
    ccpm::cca bt(&p2, rv, [] (const void *) -> bool { return true; } );
  }
  {
    ccpm::region_vector_t rv_bad(
      ccpm::region_vector_t::value_type(
        common::make_byte_span(static_cast<common::byte *>(pr)+8, size)
      )
    );
    try
    {
      ccpm::cca bt(&p2, rv_bad, [] (const void *) -> bool { return true; } );
      EXPECT_EQ(0, 1);
    }
    catch ( const std::domain_error &e )
    {
      std::cerr << "Expected failure: " << e.what() << "\n";
    }
  }
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  if (argc > 1) {
    Options.uuid = std::stoull(argv[1]);
  }
  (void) Options.uuid; // unused
  auto r = RUN_ALL_TESTS();

  return r;
}
