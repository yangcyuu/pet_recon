#include <format>
#include <fstream>

#include "../public.hpp"
#include "include/experimental/node/MichNorm.hpp"
#include "include/experimental/node/Senmaps.hpp"
using namespace openpni::experimental;
constexpr int NUM_SUBSETS = 12;
constexpr int CAL_SUBSET = 3;
static_assert(NUM_SUBSETS > CAL_SUBSET);
int main() {
  tools::cpu_threads = tools::CpuMultiThread::callWithAllThreads(tools::CpuMultiThread::CPUScheduleType::Dynamic, 1024);

  auto e180 = E180();

  auto grids = core::Grids<3>::create_by_spacing_size(core::Vector<float, 3>::create(.5f, .5f, .5f),
                                                      core::Vector<int64_t, 3>::create(320, 320, 400));

  openpni::experimental::node::MichNormalization norm(e180);
  norm.recoverFromFile("/home/ustc/pni_core/new/pni-standard-project/manual-test/Normalization/test/norm_factors.dat");

  node::MichSenmap senmap(1.0, e180);
  senmap.setSubsetNum(NUM_SUBSETS);
  senmap.bindNormalization(&norm);
  senmap.setPreferredSource(node::MichSenmap::SenmapSource::Senmap_CPU);

  for (int subset = 0; subset < CAL_SUBSET; subset++) {
    auto ptr = senmap.dumpHSenmap(subset, grids);
    write_to_file(std::format("senmap_subset_{}_cpu_norm.bin", subset), ptr.get(), grids.totalSize());
    std::cout << std::format("Dump senmap_subset_{}_cpu_norm.bin done.\n", subset);
  }

  senmap.clearCache();
  senmap.setPreferredSource(node::MichSenmap::SenmapSource::Senmap_GPU);
  for (int subset = 0; subset < CAL_SUBSET; subset++) {
    auto ptr = senmap.dumpHSenmap(subset, grids);
    write_to_file(std::format("senmap_subset_{}_gpu_norm.bin", subset), ptr.get(), grids.totalSize());
    std::cout << std::format("Dump senmap_subset_{}_gpu_norm.bin done.\n", subset);
  }
}