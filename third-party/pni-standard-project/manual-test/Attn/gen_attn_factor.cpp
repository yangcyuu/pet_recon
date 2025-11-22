#include "../public.hpp"
#include "include/experimental/core/Mich.hpp"
#include "include/experimental/node/LORBatch.hpp"
#include "include/experimental/node/MichAttn.hpp"
#include "include/experimental/tools/Parallel.hpp"
using namespace openpni::experimental;

// void generateAttnE180() {
//   std::string in_attnMap = "/media/ustc-pni/4E8CF2236FB7702F/LGXTest/Data/20250826_recontestdata/FIN_WellCounter/"
//                            "CTRecon4888/ct_attn_img.dat";

//   auto e180 = E180();
//   auto michInfoHub = core::MichInfoHub::create(e180);
//   auto grids = core::Grids<3>::create_by_spacing_size(core::Vector<float, 3>::create(.5f, .5f, .5f),
//                                                       core::Vector<int64_t, 3>::create(320, 320, 400));

//   auto attnMapData = read_from_file<float>(in_attnMap, grids.totalSize(), 6);

//   // this is a example of generate attn factors by batch
//   auto runByBatch = [&](node::MichAttn::AttnFactorSource source, std::string fileName) // source is either CPU or GPU
//   {
//     std::cout << "Running " << fileName << " version by batch , " << std::endl;
//     node::MichAttn attn(e180);
//     attn.setPreferredSource(source);
//     attn.setMapSize(grids);
//     attn.bindHAttnMap(attnMapData.get());
//     node::LORBatch lorBatch(e180);
//     lorBatch.setSubsetNum(12);
//     lorBatch.setBinCut(0);
//     for (auto lors = lorBatch.setCurrentSubset(0).nextHBatch(); !lors.empty(); lors = lorBatch.nextHBatch()) {
//       auto attnFactors = attn.getHAttnFactorsBatch(lors);
//       write_to_file(std::format("attn_factors_{}_subset0_{}.dat", fileName, lors.size()), attnFactors, lors.size());
//       std::cout << "subset 0, batch size: " << lors.size() << " done\n";
//       break;
//     }
//   };
//   // this is a example of generate attn factors by mich(all lors)
//   auto run = [&](node::MichAttn::AttnFactorSource source, std::string fileName) // source is either CPU or GPU
//   {
//     node::MichAttn attn(e180);
//     attn.setPreferredSource(source);
//     attn.setMapSize(grids);
//     attn.bindHAttnMap(attnMapData.get());
//     auto attnMich = attn.dumpAttnMich();
//     write_to_file(std::format("attn_factors_{}_subset0_{}.dat", fileName, michInfoHub.getMichSize()), attnMich,
//                   michInfoHub.getMichSize());
//   };

//   std::cout << "Running GPU version" << std::endl;
//   runByBatch(node::MichAttn::Attn_GPU, "gpu"); // GPU by batch
//   runByBatch(node::MichAttn::Attn_CPU, "cpu"); // CPU by batch
//   run(node::MichAttn::Attn_GPU, "gpu_all");    // GPU all mich
//   run(node::MichAttn::Attn_CPU, "cpu_all");    // CPU all mich
// }

int main() {
  tools::cpu_threads().setThreadNumType(tools::MAX_THREAD).setScheduleType(tools::DYNAMIC).setScheduleNum(64);
  std::cout << "\n=== Mich Norm Factor Generation Test ===\n" << std::endl;
  //==============================================================CONFIG LOADING
  std::string polygonDefineFilePath = "config/polygonSystemDefine.json";
  std::string attnParamsFilePath = "config/attnParams.json";
  std::unique_ptr<float[]> HUMapData;
  std::unique_ptr<float[]> attnMapData;

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
  auto michInfo = core::MichInfoHub::create(mich);

  core::Grids<3> attnGrids;
  auto attnJson = readFromJson<openpni::autogen::json::AttnParams>(attnParamsFilePath);
  {
    attnGrids = core::Grids<3>::create_by_spacing_size(
        core::Vector<float, 3>::create(attnJson.UmapGridsX, attnJson.UmapGridsY, attnJson.UmapGridsZ),
        core::Vector<int64_t, 3>::create(attnJson.UmapGridsNumX, attnJson.UmapGridsNumY, attnJson.UmapGridsNumZ));
  }
  std::cout << "\n=== Configuration loaded successfully! ===\n" << std::endl;
  //==============================================================MAIN
  node::MichAttn attn(mich);
  attn.setMapSize(attnGrids);
  if (attnJson.UseHUMap) {
    std::cout << "Generating Attenuation Factors using HU Map..." << std::endl;
    std::string in_HUMap = "Data/source/HUMap.dat";
    HUMapData = read_from_file<float>(in_HUMap, attnGrids.totalSize(), 6);
    attn.bindHHUMap(HUMapData.get()); // bind HU map data here
  } else {
    std::cout << "Generating Attenuation Factors using Attenuation Map..." << std::endl;
    std::string in_attnMap = "Data/source/AttnMap.dat";
    attnMapData = read_from_file<float>(in_attnMap, attnGrids.totalSize(), 6);
    attn.bindHAttnMap(attnMapData.get()); // bind attn map data here
  }

  if (attnJson.UseCUDA) {
    std::cout << "Using GPU for attenuation factor calculation." << std::endl;
    attn.setPreferredSource(node::MichAttn::Attn_GPU);
  } else {
    std::cout << "Using CPU for attenuation factor calculation." << std::endl;
    attn.setPreferredSource(node::MichAttn::Attn_CPU);
  }
  auto attnFactors = attn.dumpAttnMich();
  write_to_file("Data/result/attn_factors.dat", attnFactors, michInfo.getMichSize());
  std::cout << "\n=== Attenuation factor generation completed successfully! ===\n" << std::endl;

  return 0;
}