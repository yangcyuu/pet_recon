#include <iostream>

#include "../public.hpp"
#include "include/experimental/example/SimplyRecon.hpp"
#include "include/experimental/node/GaussConv3D.hpp"
const char *fileMichName = "shell_fwd.bin";
// const char *fileMichName = "/home/ustc/pni_test/0919test/testdata/coin_all.image3d";
const int fileMichOffset = 0;
// const int fileMichOffset = 6;
using namespace openpni::experimental;
int16_t calTOFDevFormCTR(
    int16_t CTR) {
  return CTR / 2.355f / 1.4142f; // FWHM to stddev, divide sqrt(2)
}
int16_t calTOFDevFormUnitDev(
    int16_t UnitDev) {
  return UnitDev / 1.4142f; // UnitSigma = sqrt(sigma1^2 + sigma2^2), here sigma1=sigma2
}

int main() {
  auto e180 = E180();
  auto michInfo = core::MichInfoHub::create(e180);
  std::string listmode_path = "/media/ustc-pni/4E8CF2236FB7702F/LGXTest/Data/E180_lm/output.lm";
  auto michNorm = nullptr;
  auto michRand = nullptr;
  auto michScat = nullptr;
  auto michAttn = nullptr;
  std::cout << "init listmode data done\n";

  example::OSEM_TOF_params params;
  params.size_GB = 1;
  params.binCutRatio = 0;
  params.iterNum = 1;
  params.sample_rate = 0.5f;
  params.subsetNum = 12;
  params.TOF_division = 1000;
  auto conv3D = node::GaussianConv3D();
  conv3D.setHWHM(1.0f);
  core::Grids<3, float> grids = core::Grids<3, float>::create_by_spacing_size(
      core::Vector<float, 3>::create(.5f, .5f, .5f), core::Vector<int64_t, 3>::create(320, 320, 400));
  std::unique_ptr<float[]> outImg = std::make_unique_for_overwrite<float[]>(grids.totalSize());
  example::instant_OSEM_listmodeTOF_CUDA(core::Image3DOutput<float>{grids, outImg.get()}, params, conv3D, listmode_path,
                                         michNorm, michRand, michScat, michAttn, e180);
  std::cout << "OSEM done\n";
  std::ofstream outFile("recon_img_listmodeTOF_gpu.bin", std::ios::binary);
  outFile.write(reinterpret_cast<char *>(outImg.get()), grids.totalSize() * sizeof(float));
  outFile.close();
}