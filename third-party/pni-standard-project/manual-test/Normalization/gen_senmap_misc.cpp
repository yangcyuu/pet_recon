#include <format>
#include <fstream>

#include "../public.hpp"
#include "include/experimental/node/GaussConv3D.hpp"
#include "include/experimental/node/MichNorm.hpp"
#include "include/experimental/node/Senmaps.hpp"
#include "include/experimental/tools/Parallel.hpp"
using namespace openpni::experimental;
constexpr int NUM_SUBSETS = 12;
constexpr int CAL_SUBSET = 12;
static_assert(NUM_SUBSETS >= CAL_SUBSET);
int main() {
  tools::cpu_threads().setThreadNumType(tools::MAX_THREAD).setScheduleType(tools::DYNAMIC).setScheduleNum(64);

  auto __930 = _930();

  core::Grids<3, float> grids = core::Grids<3, float>::create_by_spacing_size(
      core::Vector<float, 3>::create(2.0f, 2.0f, 2.1f), core::Vector<int64_t, 3>::create(250, 250, 144));

  openpni::experimental::node::MichNormalization norm(__930);
  norm.recoverFromFile("/media/lenovo/1TB/a_new_envir/v4_coin/data/save/930IQ/pni_mich_norm_file.bin");

  node::GaussianConv3D conv3D;
  conv3D.setHWHM(2.0f);
  node::MichSenmap senmap(conv3D, __930);
  senmap.setSubsetNum(NUM_SUBSETS);
  senmap.bindNormalization(&norm);
  senmap.setPreferredSource(node::MichSenmap::SenmapSource::Senmap_GPU);

  auto range = std::views::iota(0, CAL_SUBSET) | std::views::filter([&](int subset) { return subset % 3 == 0; }) |
               std::views::transform([&](int subset) { return std::make_pair(grids, subset); });
  senmap.preBaking(std::vector<std::pair<core::Grids<3, float>, int>>(range.begin(), range.end()));

  for (int subset = 0; subset < CAL_SUBSET; subset++) {
    auto ptr = senmap.dumpHSenmap(subset, grids);
    write_to_file(std::format("senmap_subset_{}_gpu.bin", subset), ptr.get(), grids.totalSize());
  }
}