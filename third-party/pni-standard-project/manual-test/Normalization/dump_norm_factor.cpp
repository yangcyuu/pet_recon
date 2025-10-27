#include <format>
#include <fstream>

#include "../public.hpp"
#include "src/experimental/impl/MichNormImpl.hpp"
using namespace openpni::experimental;

int main() {
  tools::cpu_threads().setThreadNumType(tools::MAX_THREAD).setScheduleType(tools::DYNAMIC).setScheduleNum(64);

  auto __930 = _930();

  auto michInfoHub = core::MichInfoHub::create(__930);

  openpni::experimental::node::MichNormalization_impl norm(__930);
  norm.recoverFromFile("/media/lenovo/1TB/a_new_envir/v4_coin/data/save/930IQ/pni_mich_norm_file.bin");

  std::cout << "Doing normalization calculation..." << std::endl;
#define DUMP(filename, action)                                                                                         \
  {                                                                                                                    \
    write_to_file(filename, norm.action(), michInfoHub.getMichSize());                                                 \
    std::cout << std::format("Dump {} done.\n", filename);                                                             \
  }
  DUMP("norm_mich.bin", dumpNormalizationMich);
  DUMP("norm_cryfct.bin", dumpCryFctMich);
  DUMP("norm_blockfct.bin", dumpBlockFctMich);
  DUMP("norm_radialfct.bin", dumpRadialFctMich);
  DUMP("norm_planefct.bin", dumpPlaneFctMich);
  DUMP("norm_interferencefct.bin", dumpInterferenceFctMich);
  DUMP("norm_dtcomponent.bin", dumpDTComponentMich);

  std::cout << "Save to norm_factors.dat done." << std::endl;
#undef DUMP
}