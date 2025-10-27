#include <format>
#include <fstream>

#include "../public.hpp"
#include "include/experimental/node/Senmaps.hpp"
using namespace openpni::experimental;
constexpr int NUM_SUBSETS = 12;
constexpr int CAL_SUBSET = 1;
static_assert(NUM_SUBSETS > CAL_SUBSET);
int main() {
  tools::cpu_threads().setThreadNumType(tools::MAX_THREAD).setScheduleType(tools::DYNAMIC).setScheduleNum(64);

  auto e180 = E180();

  core::Grids<3, float> grids = core::Grids<3>::create_by_spacing_size(core::Vector<float, 3>::create(.5f, .5f, .5f),

                                                                       core::Vector<int64_t, 3>::create(320, 320, 400));
  node::MichSenmap senmap(1.0, e180);
  senmap.setSubsetNum(NUM_SUBSETS);
  // senmap.setPreferredSource(node::MichSenmap::SenmapSource::Senmap_CPU);

  // for (int subset = 0; subset < CAL_SUBSET; subset++) {
  //   auto ptr = senmap.dumpHSenmap(subset, grids);
  //   write_to_file(std::format("senmap_subset_{}_cpu.bin", subset), ptr.get(), grids.totalSize());
  //   std::cout << std::format("Dump senmap_subset_{}_cpu.bin done.\n", subset);
  // }

  senmap.clearCache();
  senmap.setPreferredSource(node::MichSenmap::SenmapSource::Senmap_GPU);
  for (int subset = 0; subset < CAL_SUBSET; subset++) {
    auto ptr = senmap.dumpHSenmap(subset, grids);
    write_to_file(std::format("senmap_subset_{}_gpu.bin", subset), ptr.get(), grids.totalSize());
    std::cout << std::format("Dump senmap_subset_{}_gpu.bin done.\n", subset);
  }
}