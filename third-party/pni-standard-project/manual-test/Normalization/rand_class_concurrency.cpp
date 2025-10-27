#include <algorithm>
#include <format>
#include <fstream>
#include <numeric>
#include <semaphore>
#include <thread>
#include <vector>

#include "../public.hpp"
#include "include/experimental/node/Random.hpp"
constexpr int minSectorDifference{4};
constexpr int radialModuleNumS{6};
constexpr float BadChannelThreshold{0.02};

constexpr int max_thread_concurrent{16};
constexpr int test_count{100};

int main() {
  std::string in_delayFile = "/home/ustc/LGX_TEST/Data/20250826_recontestdata/FIN_WellCounter/"
                             "Slices_of_PET4643_2025-04-10-14-59-57/Slice4796/delay_all.image3d";
  std::string out_randFile = "randCorrection.bin";
  auto e180 = E180();

  auto delayMich = read_from_file<float>(in_delayFile, e180.michInfoHub().getMichSize(), 6);

  openpni::experimental::node::RandomFromMich random(e180);
  random.setBadChannelThreshold(BadChannelThreshold);
  random.setRadialModuleNumS(radialModuleNumS);
  random.setMinSectorDifference(minSectorDifference);
  random.setDelayMich(delayMich.get());

  std::cout << "Begin concurrency test..." << std::endl;
  std::counting_semaphore<max_thread_concurrent> semaphore(max_thread_concurrent);
  std::vector<std::jthread> threads;
  for (const auto _ : std::views::iota(0, test_count))
    threads.push_back(std::jthread([&] mutable -> void {
      semaphore.acquire();
      auto t_random = random.copy();
      auto ptr = t_random.dumpFactorsAsHMich();
      auto sum = std::accumulate(ptr.get(), ptr.get() + e180.michInfoHub().getMichSize(), 0.0);
      std::cout << std::format("Thread ID: {}, \tSum of factors: {}\n",
                               std::hash<std::thread::id>{}(std::this_thread::get_id()), sum);
      semaphore.release();
    }));
  std::cout << "Doing random calculation..." << std::endl;
  threads.clear();
  std::cout << "Random calculation done." << std::endl;
}