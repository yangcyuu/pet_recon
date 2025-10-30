#include <format>
#include <iostream>

#include "../public.hpp"
#include "src/experimental/impl/ShellMichHelper.hpp"
using namespace openpni::experimental;

int main() {
  tools::cpu_threads().setThreadNumType(tools::MAX_THREAD).setScheduleType(tools::DYNAMIC).setScheduleNum(64);

  auto mich = E180_shell();
  node::impl::ShellMichHelper helper(mich);
  helper.setShellSize(75, 80, 1000);
  core::Grids<3, float> grids = decltype(grids)::create_by_center_spacing_size(
      core::Vector<float, 3>::create(0, 0, 0), core::Vector<float, 3>::create(0.5, 0.5, 0.5),
      core::Vector<int64_t, 3>::create(340, 340, 440));
  helper.setGrids(grids);

  const auto lorNumPerSlice =
      core::MichInfoHub::create(mich).getBinNum() * core::MichInfoHub::create(mich).getViewNum();
  std::fstream file("shell_fwd.bin", std::ios::out | std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("can not open file shell_fwd.bin");
  for (const auto slice : std::views::iota(0u, core::MichInfoHub::create(mich).getSliceNum())) {
    auto data = helper.getOneMichSlice(slice);
    file.write(reinterpret_cast<const char *>(data), sizeof(float) * lorNumPerSlice);
    if (slice % 100 == 0)
      std::cout << " slice " << slice << " done\n";
  }
}