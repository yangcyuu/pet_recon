#include "../public.hpp"
#include "include/experimental/core/Mich.hpp"
#include "include/experimental/node/LORBatch.hpp"
#include "include/experimental/node/MichAttn.hpp"
#include "include/experimental/tools/Parallel.hpp"
using namespace openpni::experimental;

void generateAttnE180() {
  std::string in_attnMap = "/media/ustc-pni/4E8CF2236FB7702F/LGXTest/Data/20250826_recontestdata/FIN_WellCounter/"
                           "CTRecon4888/ct_attn_img.dat";

  auto e180 = E180();
  auto michInfoHub = core::MichInfoHub::create(e180);
  auto grids = core::Grids<3>::create_by_spacing_size(core::Vector<float, 3>::create(.5f, .5f, .5f),
                                                      core::Vector<int64_t, 3>::create(320, 320, 400));

  auto attnMapData = read_from_file<float>(in_attnMap, grids.totalSize(), 6);

  // this is a example of generate attn factors by batch
  auto runByBatch = [&](node::MichAttn::AttnFactorSource source, std::string fileName) // source is either CPU or GPU
  {
    std::cout << "Running " << fileName << " version by batch , " << std::endl;
    node::MichAttn attn(e180);
    attn.setPreferredSource(source);
    attn.setMapSize(grids);
    attn.bindHAttnMap(attnMapData.get());
    node::LORBatch lorBatch(e180);
    lorBatch.setSubsetNum(12);
    lorBatch.setBinCut(0);
    for (auto lors = lorBatch.setCurrentSubset(0).nextHBatch(); !lors.empty(); lors = lorBatch.nextHBatch()) {
      auto attnFactors = attn.getHAttnFactorsBatch(lors);
      write_to_file(std::format("attn_factors_{}_subset0_{}.dat", fileName, lors.size()), attnFactors, lors.size());
      std::cout << "subset 0, batch size: " << lors.size() << " done\n";
      break;
    }
  };
  // this is a example of generate attn factors by mich(all lors)
  auto run = [&](node::MichAttn::AttnFactorSource source, std::string fileName) // source is either CPU or GPU
  {
    node::MichAttn attn(e180);
    attn.setPreferredSource(source);
    attn.setMapSize(grids);
    attn.bindHAttnMap(attnMapData.get());
    auto attnMich = attn.dumpAttnMich();
    write_to_file(std::format("attn_factors_{}_subset0_{}.dat", fileName, michInfoHub.getMichSize()), attnMich,
                  michInfoHub.getMichSize());
  };

  std::cout << "Running GPU version" << std::endl;
  runByBatch(node::MichAttn::Attn_GPU, "gpu"); // GPU by batch
  runByBatch(node::MichAttn::Attn_CPU, "cpu"); // CPU by batch
  run(node::MichAttn::Attn_GPU, "gpu_all");    // GPU all mich
  run(node::MichAttn::Attn_CPU, "cpu_all");    // CPU all mich
}

int main() {
  tools::cpu_threads().setThreadNumType(tools::MAX_THREAD).setScheduleType(tools::DYNAMIC).setScheduleNum(64);
  generateAttnE180();

  return 0;
}