#include <format>
#include <fstream>

#include "../public.hpp"
#include "include/experimental/node/MichDeadTime.hpp"
#include "include/experimental/tools/Parallel.hpp"

using namespace openpni::experimental;

void generateE180CFDT() {

  auto polySys = E180();
  auto michInfoHub = core::MichInfoHub::create(polySys);
  openpni::experimental::node::MichDeadTime deadTime(polySys);

  // experiment data，prompt and delay
  std::vector<float> test_prompt1(michInfoHub.getMichSize(), 1.f);
  std::vector<float> test_prompt2(michInfoHub.getMichSize(), 2.f);
  std::vector<float> test_prompt3(michInfoHub.getMichSize(), 3.f);

  std::vector<float> test_delay1(michInfoHub.getMichSize(), 0.1f);
  std::vector<float> test_delay2(michInfoHub.getMichSize(), 0.2f);
  std::vector<float> test_delay3(michInfoHub.getMichSize(), 0.3f);
  // append this with scanTime and activity

  deadTime.appendAcquisition(test_prompt1.data(), test_delay1.data(), 600, 196.7317847);
  deadTime.appendAcquisition(test_prompt2.data(), test_delay2.data(), 300, 77.28871293);
  deadTime.appendAcquisition(test_prompt3.data(), test_delay3.data(), 100, 2.556896);

  auto cfdt = deadTime.dumpCFDTTable();
  write_to_file("test_cfdt.bin", cfdt.data(), cfdt.size());
  auto rt = deadTime.dumpRTTable();
  write_to_file("test_rt.bin", rt.data(), rt.size());

  deadTime.dumpToFile("test_deadtime.dat");
}

int main() {
  tools::cpu_threads().setThreadNumType(tools::MAX_THREAD).setScheduleType(tools::DYNAMIC).setScheduleNum(64);
  generateE180CFDT();
  return 0;
}