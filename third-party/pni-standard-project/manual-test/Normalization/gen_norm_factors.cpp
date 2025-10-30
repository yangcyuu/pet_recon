#include <format>
#include <fstream>

#include "../public.hpp"
#include "src/experimental/impl/MichNormImpl.hpp"
using namespace openpni::experimental;
void generateE180NormFactors() {
  tools::cpu_threads().setThreadNumType(tools::MAX_THREAD).setScheduleType(tools::DYNAMIC).setScheduleNum(64);

  std::string in_normFile =
      "/media/ustc-pni/4E8CF2236FB7702F/LGXTest/Data/20250826_recontestdata/normSlice4652/coin_all.image3d";
  // std::string in_fwdFile = "/media/ustc-pni/4E8CF2236FB7702F/LGXTest/test_0919/v3normFWD.bin";

  auto e180 = E180();
  auto michInfoHub = core::MichInfoHub::create(e180);
  auto normData = read_from_file<float>(in_normFile, michInfoHub.getMichSize(), 6);
  std::cout << std::format("read norm data done, size = {}\n", michInfoHub.getMichSize());
  // auto fwdData = read_from_file<float>(in_fwdFile, michInfoHub.getMichSize(), 0);
  // std::cout << std::format("read fwd data done, size = {}\n", michInfoHub.getMichSize());

  // norm param
  float fActCorrCutLow{0.05};
  float fActCorrCutHigh{0.22};
  float fCoffCutLow{0.0};
  float fCoffCutHigh{100.0};
  float BadChannelThreshold{0.02};
  int radialModuleNumS{4};

  openpni::experimental::node::MichNormalization_impl norm(e180);
  norm.setBadChannelThreshold(BadChannelThreshold);
  norm.setFActCorrCutLow(fActCorrCutLow);
  norm.setFActCorrCutHigh(fActCorrCutHigh);
  norm.setFCoffCutLow(fCoffCutLow);
  norm.setFCoffCutHigh(fCoffCutHigh);
  norm.setRadialModuleNumS(radialModuleNumS);
  norm.bindShellScanMich(normData.get());
  // norm.bindShellFwdMich(fwdData.get());
  norm.setShell( // 设置壳体参数
      75.f,      // 壳体内半径，单位mm
      80.f,      // 壳体外半径，单位mm
      1000.f,    // 壳体轴向长度，单位mm
      114.0f,    // 视差效应下的扫描仪半径
      core::Grids<3, float>::create_by_spacing_size(core::Vector<float, 3>::create(0.5, 0.5, 0.5),
                                                    core::Vector<int64_t, 3>::create(340, 340, 440)));
  auto data = norm.getActivityMich(); // 计算壳体的正向投影
  std::ofstream outFile("fwd.bin", std::ios::binary);
  outFile.write(reinterpret_cast<char *>(data.get()), MichInfoHub(e180).getMichSize() * sizeof(float));
  outFile.close();

  std::cout << "Doing normalization calculation..." << std::endl;
#define DUMP(filename, action)                                                                                         \
  {                                                                                                                    \
    write_to_file(filename, norm.action(), michInfoHub.getMichSize());                                                 \
    std::cout << std::format("Dump {} done.\n", filename);                                                             \
  }
  DUMP("norm_mich_new.bin", dumpNormalizationMich);
  DUMP("norm_cryfct_new.bin", dumpCryFctMich);
  DUMP("norm_blockfct_new.bin", dumpBlockFctMich);
  DUMP("norm_radialfct_new.bin", dumpRadialFctMich);
  DUMP("norm_planefct_new.bin", dumpPlaneFctMich);
  DUMP("norm_interferencefct_new.bin", dumpInterferenceFctMich);
  DUMP("norm_dtcomponent_new.bin", dumpDTComponentMich);
  std::cout << "Normalization calculation done." << std::endl;

  norm.saveToFile("norm_factors.dat");
  std::cout << "Save to norm_factors.dat done." << std::endl;
#undef DUMP
}

// int main() {
//   std::string in_normFile =
//       "/media/ustc-pni/4E8CF2236FB7702F/LGXTest/Data/20250826_recontestdata/normSlice4652/coin_all.image3d";
//   std::string in_fwdFile = "/media/ustc-pni/4E8CF2236FB7702F/LGXTest/test_0919/v3normFWD.bin";

//   auto e180 = E180();

//   auto michInfoHub = core::MichInfoHub::create(e180);
//   auto normData = read_from_file<float>(in_normFile, michInfoHub.getMichSize(), 6);
//   std::cout << std::format("read norm data done, size = {}\n", michInfoHub.getMichSize());
//   auto fwdData = read_from_file<float>(in_fwdFile, michInfoHub.getMichSize(), 0);
//   std::cout << std::format("read fwd data done, size = {}\n", michInfoHub.getMichSize());

//   // norm param
//   float fActCorrCutLow{0.05};
//   float fActCorrCutHigh{0.22};
//   float fCoffCutLow{0.0};
//   float fCoffCutHigh{100.0};
//   float BadChannelThreshold{0.02};
//   int radialModuleNumS{4};

//   openpni::experimental::node::MichNormalization_impl norm(e180);
//   norm.setBadChannelThreshold(BadChannelThreshold);
//   norm.setFActCorrCutLow(fActCorrCutLow);
//   norm.setFActCorrCutHigh(fActCorrCutHigh);
//   norm.setFCoffCutLow(fCoffCutLow);
//   norm.setFCoffCutHigh(fCoffCutHigh);
//   norm.setRadialModuleNumS(radialModuleNumS);
//   norm.bindShellScanMich(normData.get());
//   norm.bindShellFwdMich(fwdData.get());

//   std::cout << "Doing normalization calculation..." << std::endl;
// #define DUMP(filename, action) \
//   { \
//     write_to_file(filename, norm.action(), michInfoHub.getMichSize()); \
//     std::cout << std::format("Dump {} done.\n", filename); \
//   }
//   DUMP("norm_mich.bin", dumpNormalizationMich);
//   DUMP("norm_cryfct.bin", dumpCryFctMich);
//   DUMP("norm_blockfct.bin", dumpBlockFctMich);
//   DUMP("norm_radialfct.bin", dumpRadialFctMich);
//   DUMP("norm_planefct.bin", dumpPlaneFctMich);
//   DUMP("norm_interferencefct.bin", dumpInterferenceFctMich);
//   DUMP("norm_dtcomponent.bin", dumpDTComponentMich);
//   std::cout << "Normalization calculation done." << std::endl;

//   norm.saveToFile("norm_factors.dat");
//   std::cout << "Save to norm_factors.dat done." << std::endl;
// #undef DUMP
// }

int main() {
  generateE180NormFactors();
  return 0;
}