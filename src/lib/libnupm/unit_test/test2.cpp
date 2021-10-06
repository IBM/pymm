#include "region_modifications.h"
#include "allocator_ra.h"
#include "rc_alloc_lb.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <memory>

//#define GPERF_TOOLS

#ifdef GPERF_TOOLS
#include <gperftools/profiler.h>
#endif

// The fixture for testing class Libnupm.
class Libnupm_test : public ::testing::Test
{
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

TEST_F(Libnupm_test, RegionModifications)
{
  /* Note: the nupm::region_tracker_add function uses a thread_local
   * Region_modifications object, not the one declared below.
   * However, calls to region_tracker_add are ignored unlessi
   * nupm::tracker_active is true. The constructor of r sets it true,
   * and the destructor of r returns it to false, providing a scope
   * in which region_tracker_add is effective.
   */
  nupm::Region_modifications r;
  struct
  {
    char c = 0;
    /* padding here */
    int i = 2;
    int j = 2;
  } z;
  nupm::region_tracker_add(&z.j, sizeof z.j);
  nupm::region_tracker_add(&z.c, sizeof z.c);
  nupm::region_tracker_add(&z.i, sizeof z.i);

  const void *v;
  std::size_t sz;
  sz = nupm::region_tracker_get_region(1, v);
  EXPECT_EQ(sizeof z.i + sizeof z.j, sz);
  EXPECT_EQ(&z.i, v);
  sz = nupm::region_tracker_get_region(0, v);
  EXPECT_EQ(sizeof z.c, sz);
  EXPECT_EQ(&z.c, v);
  /* past last element */
  sz = nupm::region_tracker_get_region(3, v);
  EXPECT_EQ(0, sz);
  /* farther past last element */
  sz = nupm::region_tracker_get_region(5, v);
  EXPECT_EQ(0, sz);
  /* before first element: also nothing */
  sz = nupm::region_tracker_get_region(offset_t(-1), v);
  EXPECT_EQ(0, sz);
  nupm::region_tracker_coalesce_across_TLS();
  sz = nupm::region_tracker_get_region(1, v);
  EXPECT_EQ(sizeof z.i + sizeof z.j, sz);
  EXPECT_EQ(&z.i, v);
  nupm::region_tracker_clear();
  sz = nupm::region_tracker_get_region(0, v);
  EXPECT_EQ(0, sz);
}

TEST_F(Libnupm_test, AVL_allocator)
{
  auto size = 1000000U;
  void *v = malloc(size);
  ASSERT_NE(nullptr, v);
  /* AVL_range_allocator requires an addr_t, defined in comanche common/types.h */
  core::AVL_range_allocator ra(reinterpret_cast<addr_t>(v), size);
  auto a = nupm::allocator_ra<char>(ra);
  const std::size_t ITERATIONS = 100;
  PLOG("AVL_allocator running... %zu allocations", ITERATIONS);

  std::vector<char *> alloc_v;
  alloc_v.reserve(ITERATIONS);

  auto start = std::chrono::high_resolution_clock::now();
  constexpr auto alloc_size = 200;
  for (unsigned i = 0; i < ITERATIONS; i++)
  {
    auto p = a.allocate(alloc_size);
    ASSERT_TRUE(p);
    EXPECT_LE(v, p);
    EXPECT_LT(p + alloc_size, static_cast<char *>(v) + size);
    alloc_v.push_back(p);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto secs = std::chrono::duration<double>(end - start).count();
  PINF("AVL_allocator: %lu allocs/sec",
       static_cast<unsigned long>(double(ITERATIONS) / secs));

  PLOG("AVL_allocator: Freeing %zu allocations...", alloc_v.size());

  for (auto &vv : alloc_v)
  {
      a.deallocate(vv, alloc_size);
  }
}

TEST_F(Libnupm_test, RCA_LB_allocator)
{
  std::size_t region_size = 10485760;
  int numa_node = 0;
  auto v0 = std::make_unique<char[]>(region_size);
  ASSERT_NE(nullptr, v0);
  auto v1 = std::make_unique<char[]>(region_size);
  ASSERT_NE(nullptr, v0);
  /* AVL_range_allocator requires an addr_t, defined in comanche common/types.h */
  nupm::Rca_LB lb(0);
  lb.add_managed_region(v0.get(), region_size, numa_node);
#if 0 /* a second area would make the EXPECT sanity checks on erturn pointers trickier */
  lb.add_managed_region(v1.get(), region_size, numa_node);
#endif
  nupm::allocator_adaptor<char, nupm::Rca_LB> a(lb);
  const std::size_t ITERATIONS = 100;
  PLOG("AVL_allocator running... %zu allocations", ITERATIONS);

  std::vector<char *> alloc_v;
  alloc_v.reserve(ITERATIONS);

  auto start = std::chrono::high_resolution_clock::now();
  constexpr auto alloc_size = 200;
  std::size_t alignment = 8;
  for (unsigned i = 0; i < ITERATIONS; i++)
  {
    auto p = a.allocate(alloc_size, alignment);
    ASSERT_TRUE(p);
    EXPECT_LE(v0.get(), p);
    EXPECT_LT(p + alloc_size, static_cast<char *>(v0.get()) + region_size);
    alloc_v.push_back(p);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto secs = std::chrono::duration<double>(end - start).count();
  PINF("AVL_allocator: %lu allocs/sec",
       static_cast<unsigned long>(double(ITERATIONS) / secs));

  PLOG("AVL_allocator: Freeing %zu allocations...", alloc_v.size());
  for (auto &vv : alloc_v)
  {
    a.deallocate(vv, alloc_size, alignment);
  }
}

#undef SHOW_ORDER

TEST_F(Libnupm_test, RCA_LB_allocator_issue155)
{
  std::size_t region_size = GB(20);
  int numa_node = 0;
  auto v0 = std::make_unique<char[]>(region_size);
  ASSERT_NE(nullptr, v0);
  /* AVL_range_allocator requires an addr_t, defined in comanche common/types.h */
  nupm::Rca_LB lb(0);
  lb.add_managed_region(v0.get(), region_size, numa_node);

  nupm::allocator_adaptor<char, nupm::Rca_LB> heap(lb);

  const std::size_t COUNT = 7; /* careful, this gets mighty big, mighty fast! */

  typedef struct { char* ptr; size_t len; size_t alignment; } info_t;

  std::vector<unsigned> rel_order;  
  for(unsigned i=0;i<COUNT;i++)
    rel_order.push_back(i);

  std::vector<unsigned> alloc_order;  
  for(unsigned i=0;i<COUNT;i++)
    alloc_order.push_back(i);

  std::sort(alloc_order.begin(), alloc_order.end());

  /* generate release permutations */
  do {

#ifdef SHOW_ORDER
    std::cout << "Allocation_Order: ";
    for(auto i : alloc_order)
      std::cout << i << " ";
    std::cout << "\n";
#endif

    std::sort(rel_order.begin(), rel_order.end());

    do { /* permutate on allocation orders */

#ifdef SHOW_ORDER
      std::cout << "Rel_Order: ";
      for(auto i : rel_order)
        std::cout << i << " ";
      std::cout << "\n";
#endif
      /* run permutation */
      std::vector<info_t> allocations;
      const std::size_t ELEMENT_SIZES[] = {1048576, 134217752, 268435480, 536870936, 1073741848};
      const std::size_t ALIGNMENT_SIZES[] = {1048576, 8, 8, 8, 8};

      for(unsigned i=0;i<COUNT;i++) {
        /* change to 5 to enable 1GB allocations */
        auto j = abs(rand()) % 5; /* randomly pick, not exhaustive but hopefully enough */
        assert(j < 5);
        allocations.push_back({heap.allocate(ELEMENT_SIZES[j], ALIGNMENT_SIZES[j]), ELEMENT_SIZES[j], ALIGNMENT_SIZES[j]});
      }

      for(auto i: rel_order) {
        auto& element = allocations[i];
        heap.deallocate(element.ptr, element.len, element.alignment);
      }
    }
    while(next_permutation(rel_order.begin(), rel_order.end()));
  }
  while(next_permutation(alloc_order.begin(), alloc_order.end()));
}


int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  auto r = RUN_ALL_TESTS();

  return r;
}
