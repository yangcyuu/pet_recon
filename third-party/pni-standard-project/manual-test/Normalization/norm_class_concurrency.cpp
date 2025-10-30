#include <format>
#include <fstream>
#include <numeric>
#include <semaphore>
#include <thread>

#include "../public.hpp"
#include "include/experimental/node/Normalization.hpp"
constexpr float fActCorrCutLow{0.05};
constexpr float fActCorrCutHigh{0.22};
constexpr float fCoffCutLow{0.0};
constexpr float fCoffCutHigh{100.0};
constexpr float BadChannelThreshold{0.02};
constexpr int radialModuleNumS{4};
constexpr int max_thread_concurrent{16};
constexpr int test_count{100};
int main() {
  std::string in_normFile = "/home/ustc/LGX_TEST/Data/20250826_recontestdata/normSlice4652/coin_all.image3d";
  std::string in_fwdFile = "/home/ustc/LGX_TEST/test_0919/v3normFWD.bin";

  auto e180 = E180();

  auto normData = read_from_file<float>(in_normFile, e180.michInfoHub().getMichSize(), 6);
  std::cout << std::format("read norm data done, size = {}\n", e180.michInfoHub().getMichSize());
  auto fwdData = read_from_file<float>(in_fwdFile, e180.michInfoHub().getMichSize(), 0);
  std::cout << std::format("read fwd data done, size = {}\n", e180.michInfoHub().getMichSize());

  openpni::experimental::node::MichNormalization norm(e180);
  norm.setBadChannelThreshold(BadChannelThreshold);
  norm.setFActCorrCutLow(fActCorrCutLow);
  norm.setFActCorrCutHigh(fActCorrCutHigh);
  norm.setFCoffCutLow(fCoffCutLow);
  norm.setFCoffCutHigh(fCoffCutHigh);
  norm.setRadialModuleNumS(radialModuleNumS);
  norm.bindComponentNormScanMich(normData.get());
  norm.bindComponentNormIdealMich(fwdData.get());

  std::cout << "Begin concurrency test..." << std::endl;
  std::counting_semaphore<max_thread_concurrent> semaphore(max_thread_concurrent);
  std::vector<std::jthread> threads;
  for (const auto _ : std::views::iota(0, test_count))
    threads.push_back(std::jthread([&] mutable -> void {
      semaphore.acquire();
      auto t_norm = norm.copy();
      auto ptr = t_norm.dumpNormalizationMich();
      auto sum = std::accumulate(ptr.get(), ptr.get() + e180.michInfoHub().getMichSize(), 0.0);
      std::cout << std::format("Thread ID: {}, Sum of factors: {}\n",
                               std::hash<std::thread::id>{}(std::this_thread::get_id()), sum);
      semaphore.release();
    }));
  threads.clear();
  std::cout << "Finish norm calculation..." << std::endl;
}