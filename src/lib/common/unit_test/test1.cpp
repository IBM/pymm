/* note: we do not include component source, only the API definition */
#include <common/cycles.h>
#include <common/mpmc_bounded_queue.h>
#include <common/rand.h>
#include <common/utils.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <thread>

//#define TEST_MPMC
#define TEST_SPSC
//#define TEST_SPSC_FAST

//#define GPERF_TOOLS

#ifdef GPERF_TOOLS
#include <gperftools/profiler.h>
#endif

class Libcommon_test : public ::testing::Test {
};

#define QUEUE_SIZE 8
#define COUNT 100

#ifdef TEST_MPMC
void mpmc_receiver(common::Mpmc_bounded_lfq<uint64_t>& queue)
{
  uint64_t x;
  for (unsigned i = 0; i < COUNT; i++) {
    x = 0;
    while (!queue.pop(x))
      ;  // if queue is empty, pop will fail
    PINF("Mpmc_Receiver got: %lu", x);
  }
  PLOG("Mpmc_Receiver thread exiting.");
}

TEST_F(Libcommon_test, mpmc_queue_test)
{
  auto buffer = common::Mpmc_bounded_lfq<uint64_t>::allocate_queue_memory(QUEUE_SIZE);

  common::Mpmc_bounded_lfq<uint64_t> queue(QUEUE_SIZE, buffer);

  std::thread forked_thread([&]() { mpmc_receiver(queue); });

  sleep(1);

  for (uint64_t i = 0; i < COUNT; i++) {
    while (!queue.push(i))
      ;  // if queue is full, push will fail
    PINF("Mpmc sent %lu", i);
  }

  forked_thread.join();
}

#endif

#ifdef TEST_SPSC_FAST
unsigned long FAST_COUNT = 10000000;

void spsc_receiver_fast(common::Spsc_bounded_lfq<uint64_t>& queue)
{
  uint64_t x;
  for (unsigned long i = 0; i < FAST_COUNT; i++) {
    x = 0;
    while (!queue.pop(x))
      ;  // if queue is empty, pop will fail
    ASSERT_TRUE(x == i);
  }
  PLOG("Spsc_Receiver thread exiting.");
}

TEST_F(Libcommon_test, spsc_queue_test_fast)
{
  auto buffer = common::Spsc_bounded_lfq<uint64_t>::allocate_queue_memory(256);

  common::Spsc_bounded_lfq<uint64_t> queue(256, buffer);

  std::thread forked_thread([&]() { spsc_receiver_fast(queue); });

  sleep(1);

  auto start_time = std::chrono::high_resolution_clock::now();

  for (uint64_t i = 0; i < FAST_COUNT; i++) {
    while (!queue.push(i))
      ;  // if queue is full, push will fail
  }

  forked_thread.join();

  auto end_time = std::chrono::high_resolution_clock::now();

  double secs = std::chrono::duration<double>(end_time - start_time).count();

  double per_sec = ((static_cast<double>(FAST_COUNT)) / secs);
  PINF("Time: %.2f sec", secs);
  PINF("Rate: %.0f /sec", per_sec);
}
#endif

TEST_F(Libcommon_test, get_rdtsc_frequency_mhz)
{
  PMAJOR("Clock frequency %f MHz", common::get_rdtsc_frequency_mhz());
}

//-------------------------------

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  auto r = RUN_ALL_TESTS();

  return r;
}
