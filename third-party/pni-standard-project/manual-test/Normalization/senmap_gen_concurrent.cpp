#include <format>
#include <fstream>
#include <future>
#include <semaphore>
#include <thread>

#include "../public.hpp"
#include "include/experimental/node/MichNorm.hpp"
#include "include/experimental/node/Senmaps.hpp"
using namespace openpni::experimental;
constexpr int NUM_SUBSETS = 12;
constexpr int GEN_CONCURRENTS = 4;
constexpr int RUN_CONCURRENTS = 200;
constexpr int RUN_ONCE_COCURRENT = 20;
auto e180 = E180();
core::Grids<3, float> grids = core::Grids<3>::create_by_spacing_size(core::Vector<float, 3>::create(.5f, .5f, .5f),
                                                                     core::Vector<int64_t, 3>::create(320, 320, 400));

std::unique_ptr<openpni::experimental::node::MichNormalization> g_norm;
std::future<std::unique_ptr<node::MichSenmap>> gen_senmap_future(
    int subsetStart, int subsetStep) {
  return std::async(std::launch::async, [=]() {
    auto senmap = std::make_unique<node::MichSenmap>(1.0, e180);
    senmap->setSubsetNum(NUM_SUBSETS);
    auto norm = g_norm->copy();
    senmap->bindNormalization(&norm);
    senmap->setPreferredSource(node::MichSenmap::SenmapSource::Senmap_GPU);

    auto range = std::views::iota(0, NUM_SUBSETS) |
                 std::views::filter([&](int subset) { return subset % subsetStep == subsetStart; }) |
                 std::views::transform([&](int subset) { return std::make_pair(grids, subset); });
    senmap->preBaking(std::vector<std::pair<core::Grids<3, float>, int>>(range.begin(), range.end()));

    return senmap;
  });
}

int main() {
  tools::cpu_threads = tools::CpuMultiThread::callWithAllThreads(tools::CpuMultiThread::CPUScheduleType::Dynamic, 1024);

  std::cout << "Loading normalization factors...\n";
  g_norm = std::make_unique<node::MichNormalization>(e180);
  g_norm->recoverFromFile("norm_factors.dat");

  std::cout << "Generating senmaps...\n";
  std::vector<std::future<std::unique_ptr<node::MichSenmap>>> futures;
  for (int i = 0; i < GEN_CONCURRENTS; ++i)
    futures.push_back(gen_senmap_future(i, GEN_CONCURRENTS));

  std::cout << "Joining senmaps...\n";
  node::MichSenmap allSenmap(1.0, e180);
  allSenmap.setSubsetNum(NUM_SUBSETS);
  allSenmap.setPreferredSource(node::MichSenmap::SenmapSource::Senmap_GPU);
  for (auto &fut : futures) {
    auto senmap = fut.get();
    allSenmap.join(senmap.get());
  }

  std::cout << "Dumping all senmaps once...\n";
  for (int subset = 0; subset < NUM_SUBSETS; subset++) {
    auto data = allSenmap.dumpHSenmap(subset, grids);
    write_to_file(std::format("senmap_subset_{}_con.raw", subset), data.get(), grids.totalSize());
  }

  std::cout << "Running concurrent senmap dumps...\n";
  std::counting_semaphore sem(RUN_ONCE_COCURRENT);
  std::vector<std::jthread> threads;
  for (int i = 0; i < RUN_CONCURRENTS; ++i) {
    threads.emplace_back([&]() mutable {
      sem.acquire();
      for (int subset = 0; subset < NUM_SUBSETS; subset++) {
        auto senmap = allSenmap.copy();
        auto data = senmap.dumpHSenmap(subset, grids);
        auto max =
            *std::max_element(data.get(), data.get() + grids.totalSize(), [](float a, float b) { return a < b; });
        std::cout << std::format("Subset {} max: {}\n", subset, max);
      }
      sem.release();
    });
  }
}