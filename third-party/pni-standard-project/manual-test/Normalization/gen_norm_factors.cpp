#include <format>
#include <fstream>

#include "../public.hpp"
#include "src/experimental/impl/MichNormImpl.hpp"
using namespace openpni::experimental;
// void generateE180NormFactors() {
//   tools::cpu_threads().setThreadNumType(tools::MAX_THREAD).setScheduleType(tools::DYNAMIC).setScheduleNum(64);

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
  // auto data = norm.getActivityMich(); // 计算壳体的正向投影
  // std::ofstream outFile("fwd.bin", std::ios::binary);
  // outFile.write(reinterpret_cast<char *>(data.get()), MichInfoHub(e180).getMichSize() * sizeof(float));
  // outFile.close();
  norm.saveToFile("norm_factors.dat");
  //   std::cout << "Doing normalization calculation..." << std::endl;
  // #define DUMP(filename, action) \
  //   { \
  //     write_to_file(filename, norm.action(), michInfoHub.getMichSize()); \
  //     std::cout << std::format("Dump {} done.\n", filename); \
  //   }
  //   DUMP("norm_mich_new.bin", dumpNormalizationMich);
  //   DUMP("norm_cryfct_new.bin", dumpCryFctMich);
  //   DUMP("norm_blockfct_new.bin", dumpBlockFctMich);
  //   DUMP("norm_radialfct_new.bin", dumpRadialFctMich);
  //   DUMP("norm_planefct_new.bin", dumpPlaneFctMich);
  //   DUMP("norm_interferencefct_new.bin", dumpInterferenceFctMich);
  //   DUMP("norm_dtcomponent_new.bin", dumpDTComponentMich);
  //   std::cout << "Normalization calculation done." << std::endl;

  //   norm.saveToFile("norm_factors.dat");
  //   std::cout << "Save to norm_factors.dat done." << std::endl;
  // #undef DUMP
}

// int main() {
//   std::string in_normFile =
//       "/media/ustc-pni/4E8CF2236FB7702F/LGXTest/Data/20250826_recontestdata/normSlice4652/coin_all.image3d";
//   // std::string in_fwdFile = "/media/ustc-pni/4E8CF2236FB7702F/LGXTest/test_0919/v3normFWD.bin";

//   auto e180 = E180();
//   auto michInfoHub = core::MichInfoHub::create(e180);
//   auto normData = read_from_file<float>(in_normFile, michInfoHub.getMichSize(), 6);
//   std::cout << std::format("read norm data done, size = {}\n", michInfoHub.getMichSize());
//   // auto fwdData = read_from_file<float>(in_fwdFile, michInfoHub.getMichSize(), 0);
//   // std::cout << std::format("read fwd data done, size = {}\n", michInfoHub.getMichSize());

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
//   // norm.bindShellFwdMich(fwdData.get());
//   norm.setShell( // 设置壳体参数
//       75.f,      // 壳体内半径，单位mm
//       80.f,      // 壳体外半径，单位mm
//       1000.f,    // 壳体轴向长度，单位mm
//       114.0f,    // 视差效应下的扫描仪半径
//       core::Grids<3, float>::create_by_spacing_size(core::Vector<float, 3>::create(0.5, 0.5, 0.5),
//                                                     core::Vector<int64_t, 3>::create(340, 340, 440)));
//   auto data = norm.dumpActivityMich(); // 计算壳体的正向投影
//   std::ofstream outFile("fwd.bin", std::ios::binary);
//   outFile.write(reinterpret_cast<char *>(data.get()), MichInfoHub(e180).getMichSize() * sizeof(float));
//   outFile.close();

//   std::cout << "Doing normalization calculation..." << std::endl;
// #define DUMP(filename, action) \
//   { \
//     write_to_file(filename, norm.action(), michInfoHub.getMichSize()); \
//     std::cout << std::format("Dump {} done.\n", filename); \
//   }
//   std::cout << "Normalization calculation done." << std::endl;

//   DUMP("norm_mich.bin", dumpNormalizationMich);
//   DUMP("norm_cryfct_mich.bin", dumpCryFctMich);
//   DUMP("norm_blockfct_mich.bin", dumpBlockFctMich);
//   DUMP("norm_radialfct_mich.bin", dumpRadialFctMich);
//   DUMP("norm_planefct_mich.bin", dumpPlaneFctMich);
//   DUMP("norm_interferencefct_mich.bin", dumpInterferenceFctMich);
//   DUMP("norm_dtcomponent_mich.bin", dumpDTComponentMich);
//   norm.saveToFile("norm_factors.dat");
//   std::cout << "Save to norm_factors.dat done." << std::endl;
// #undef DUMP

// #define DUMPVECTOR(filename, action) \
//   { \
//     write_to_file(filename, norm.action().data(), norm.action().size()); \
//     std::cout << std::format("Dump {} done.\n", filename); \
//   }
//   DUMPVECTOR("norm_cryCount.bin", dumpCryCount);
//   DUMPVECTOR("norm_blockA.bin", dumpBlockFctA);
//   DUMPVECTOR("norm_blockT.bin", dumpBlockFctT);
//   DUMPVECTOR("norm_radial.bin", dumpPlaneFct);
//   DUMPVECTOR("norm_plane.bin", dumpRadialFct);
//   DUMPVECTOR("norm_interference.bin", dumpInterferenceFct);
//   DUMPVECTOR("norm_cryFct.bin", dumpCryFct);
//   norm.saveToFile("norm_factors.dat");
//   std::cout << "Save different norm_factors.dat done." << std::endl;
// #undef DUMP
// }

// int main() {
//   std::string in_normFile = "/home/ustc/Desktop/testE180Case/Data/normScan.pni";
//   std::string in_fwdFile = "/home/ustc/Desktop/testE180Case/Data/shell_fwd.bin";

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

//   openpni::experimental::node::MichNormalization norm(e180);
//   norm.setBadChannelThreshold(BadChannelThreshold);
//   norm.setFActCorrCutLow(fActCorrCutLow);
//   norm.setFActCorrCutHigh(fActCorrCutHigh);
//   norm.setFCoffCutLow(fCoffCutLow);
//   norm.setFCoffCutHigh(fCoffCutHigh);
//   norm.setRadialModuleNumS(radialModuleNumS);
//   norm.bindComponentNormScanMich(normData.get());
//   norm.bindComponentNormIdealMich(fwdData.get());

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
//   std::cout << "Normalization calculation done." << std::endl;

//   norm.saveToFile("norm_factors.dat");
//   std::cout << "Save to norm_factors.dat done." << std::endl;
// #undef DUMP
// }

int main() {
  std::cout << "\n=== Mich Norm Factor Generation Test ===\n" << std::endl;
  //==============================================================CONFIG LOADING
  std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
  std::string polygonDefineFilePath = "config/polygonSystemDefine.json";
  std::string normParamsFilePath = "config/normalizationParams.json";
  std::string shellParamsFilePath = "config/shellParams.json";
  std::string in_shellFwdDataPath = "Data/result/shell_fwd.bin";
  std::string in_normScanDataPath = "Data/source/normScan.pni";
  std::string extendedName = "";

  openpni::experimental::core::MichDefine mich;
  {
    auto polyJson = readFromJson<openpni::autogen::json::PolygonalSystem>(polygonDefineFilePath);
    auto &polygon = mich.polygon;
    polygon.edges = polyJson.Edges;
    polygon.detectorLen = polyJson.DetectorLen;
    polygon.detectorPerEdge = polyJson.DetectorPerEdge;
    polygon.radius = polyJson.Radius;
    polygon.angleOf1stPerp = polyJson.AngleOf1stPerp;
    polygon.detectorRings = polyJson.DetectorRings;
    polygon.ringDistance = polyJson.RingDistance;
    auto &detector = mich.detector;
    detector.blockNumU = polyJson.DetectorBlockNumU;
    detector.blockNumV = polyJson.DetectorBlockNumV;
    detector.blockSizeU = polyJson.DetectorBlockSizeU;
    detector.blockSizeV = polyJson.DetectorBlockSizeV;
    detector.crystalNumU = polyJson.DetectorCrystalNumU;
    detector.crystalNumV = polyJson.DetectorCrystalNumV;
    detector.crystalSizeU = polyJson.DetectorCrystalSizeU;
    detector.crystalSizeV = polyJson.DetectorCrystalSizeV;
  }

  auto normParams = readFromJson<openpni::autogen::json::NormalizationParams>(normParamsFilePath);
  std::cout << "\n=== Configuration loaded successfully! ===\n" << std::endl;
  //==============================================================MAIN
  tools::cpu_threads().setThreadNumType(tools::MAX_THREAD).setScheduleType(tools::DYNAMIC).setScheduleNum(64);
  auto michInfo = core::MichInfoHub::create(mich);
  openpni::experimental::node::MichNormalization norm(mich);
  auto normScanData = read_from_file<float>(in_normScanDataPath, michInfo.getMichSize(), 6);
  norm.setBadChannelThreshold(normParams.BadChannelThreshold);
  norm.setFActCorrCutLow(normParams.FActCorrCutLow);
  norm.setFActCorrCutHigh(normParams.FActCorrCutHigh);
  norm.setFCoffCutLow(normParams.FCoffCutLow);
  norm.setFCoffCutHigh(normParams.FCoffCutHigh);
  norm.setRadialModuleNumS(normParams.RadialModuleNumS);
  norm.bindComponentNormScanMich(normScanData.get());
  std::unique_ptr<float[]> shellData;
  std::unique_ptr<float[]> delay;

  //===========================generate or load shell forward data
  if (ifFileExists(in_shellFwdDataPath)) {
    std::cout << "Loading exist shell forward data from file: " << in_shellFwdDataPath << std::endl;
    shellData = read_from_file<float>(in_shellFwdDataPath, michInfo.getMichSize(), 0);
    norm.bindComponentNormIdealMich(shellData.get());
  } else {
    std::cout << "No exist shell forward data file found,  generate it by config from " << shellParamsFilePath
              << std::endl;
    auto shellParams = readFromJson<openpni::autogen::json::ShellParams>(shellParamsFilePath);
    norm.setShell(                         // 设置壳体参数
        shellParams.InnerRadius,           // 壳体内半径，单位mm
        shellParams.OuterRadius,           // 壳体外半径，单位mm
        shellParams.AxialLength,           // 壳体轴向长度，单位mm
        shellParams.ParallaxScannerRadial, // 视差效应下的扫描仪半径
        core::Grids<3, float>::create_by_spacing_size(
            core::Vector<float, 3>::create(shellParams.ShellGridsX, shellParams.ShellGridsY, shellParams.ShellGridsZ),
            core::Vector<int64_t, 3>::create(shellParams.ShellGridsNumX, shellParams.ShellGridsNumY,
                                             shellParams.ShellGridsNumZ)));
    auto shellResult = norm.dumpActivityMich();                 // 计算壳体的正向投影
    std::string shellFwdDataPath = "Data/result/shell_fwd.bin"; // 预生成的 shell 数据路径
    write_to_file<float>(shellFwdDataPath, shellResult.get(), michInfo.getMichSize());
    std::cout << "shell forward data will be saved to: " << shellFwdDataPath << std::endl;
    norm.bindComponentNormIdealMich(shellResult.get());
  }
  //===========================bind selfNormalization if needed
  if (normParams.DoSelfNormalization) {
    extendedName += "_withselfNorm";
    std::string in_delayFilePath = "Data/source/delay.pni";
    if (ifFileExists(in_delayFilePath)) {
      delay = read_from_file<float>(in_delayFilePath, michInfo.getMichSize(), 6);
      norm.bindSelfNormMich(delay.get());
    } else {
      throw std::runtime_error("Self-normalization requested but delay file not found!");
    }
    std::cout << "Self-normalization delay file loaded from " << in_delayFilePath << std::endl;
  }
  //===========================bind DeadTimeCorrection if needed
  if (normParams.DoDeadTimeCorrection) {
    extendedName += "_withDTCorr";
    std::string in_dtFilePath = "Data/source/DTCalibration.pni";
    // norm.setDeadTimeTable();
  }
  //===========================generate norm factors
  std::cout << "Doing normalization calculation..." << std::endl;
  norm.saveToFile("Data/result/norm_factors" + extendedName + ".dat");
  std::cout << "\n=== norm factor generation completed successfully! ===\n" << std::endl;

#define DUMP(filename, action)                                                                                         \
  {                                                                                                                    \
    write_to_file(filename, norm.action(), michInfo.getMichSize());                                                    \
    std::cout << std::format("Dump {} done.\n", filename);                                                             \
  }
  DUMP("Data/result/norm_mich" + extendedName + ".bin", dumpNormalizationMich);
  DUMP("Data/result/norm_cryfct_mich" + extendedName + ".bin", dumpCryFctMich);
  DUMP("Data/result/norm_blockfct_mich" + extendedName + ".bin", dumpBlockFctMich);
  DUMP("Data/result/norm_radialfct_mich" + extendedName + ".bin", dumpRadialFctMich);
  DUMP("Data/result/norm_planefct_mich" + extendedName + ".bin", dumpPlaneFctMich);
  DUMP("Data/result/norm_interferencefct_mich" + extendedName + ".bin", dumpInterferenceFctMich);
#undef DUMP
#define DUMPVECTOR(filename, action)                                                                                   \
  {                                                                                                                    \
    write_to_file(filename, norm.action().data(), norm.action().size());                                               \
    std::cout << std::format("Dump {} done.\n", filename);                                                             \
  }
  DUMPVECTOR("Data/result/norm_cryCount" + extendedName + ".vec", dumpCryCount);
  DUMPVECTOR("Data/result/norm_blockA" + extendedName + ".vec", dumpBlockFctA);
  DUMPVECTOR("Data/result/norm_blockT" + extendedName + ".vec", dumpBlockFctT);
  DUMPVECTOR("Data/result/norm_radial" + extendedName + ".vec", dumpRadialFct);
  DUMPVECTOR("Data/result/norm_plane" + extendedName + ".vec", dumpPlaneFct);
  DUMPVECTOR("Data/result/norm_interference" + extendedName + ".vec", dumpInterferenceFct);
  DUMPVECTOR("Data/result/norm_cryFct" + extendedName + ".vec", dumpCryFct);
  std::cout << "Save different norm_factors.dat done." << std::endl;
#undef DUMP
}
