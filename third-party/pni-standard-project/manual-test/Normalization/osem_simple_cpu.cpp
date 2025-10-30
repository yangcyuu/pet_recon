#include <iostream>

#include "../public.hpp"
#include "include/experimental/example/SimplyRecon.hpp"
#include "include/experimental/node/GaussConv3D.hpp"
#include "include/experimental/tools/Parallel.hpp"
const char *fileMichName = "shell_fwd.bin";
const int fileMichOffset = 0;
using namespace openpni::experimental;
int main() {
  tools::cpu_threads().setThreadNumType(tools::MAX_THREAD).setScheduleType(tools::DYNAMIC).setScheduleNum(64);

  auto e180 = E180();
  auto michInfo = core::MichInfoHub::create(e180);
  auto michData = read_from_file<float>(fileMichName, michInfo.getMichSize(), fileMichOffset);
  auto michNorm = nullptr;
  auto michRand = nullptr;
  auto michScat = nullptr;
  std::cout << "read mich data done\n";

  example::OSEM_params params;
  params.binCutRatio = 0.2;
  params.iterNum = 1;
  params.sample_rate = 0.5;
  params.subsetNum = 12;
  auto conv3D = node::GaussianConv3D();
  conv3D.setHWHM(1.0f);
  core::Grids<3, float> grids = core::Grids<3, float>::create_by_spacing_size(
      core::Vector<float, 3>::create(.5f, .5f, .5f), core::Vector<int64_t, 3>::create(320, 320, 400));
  std::unique_ptr<float[]> outImg = std::make_unique_for_overwrite<float[]>(grids.totalSize());
  example::instant_OSEM_mich_CPU(core::Image3DOutput<float>{grids, outImg.get()}, params, conv3D, michData.get(),
                                 michNorm, michRand, michScat, e180);
  std::cout << "OSEM done\n";
  std::ofstream outFile("shell_recon_img_cpu.bin", std::ios::binary);
  outFile.write(reinterpret_cast<char *>(outImg.get()), grids.totalSize() * sizeof(float));
  outFile.close();
}