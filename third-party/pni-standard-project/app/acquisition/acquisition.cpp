#include "include/process/Acquisition.hpp"

#include <chrono>
#include <cxxopts.hpp>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

#include "../print.hpp"
#include "include/io/IO.hpp"
namespace cmd = cxxopts;
namespace fs = std::filesystem;
using namespace openpni::process;
using namespace std::chrono;
constexpr auto GIB = 1024ull * 1024ull * 1024ull;
std::string clock(
    uint64_t t) {
  return std::to_string(t / 1000) + "," + (t % 1000 < 100 ? "0" : "") + (t % 1000 < 10 ? "0" : "") +
         std::to_string(t % 1000);
}
uint32_t ipStr2ipInt(
    const std::string &ipStr) noexcept {
  std::regex ipRegex(R"((\d+)\.(\d+)\.(\d+)\.(\d+))");
  std::smatch match;
  if (std::regex_match(ipStr, match, ipRegex) && match.size() == 5) {
    try {
      uint32_t ipInt = 0;
      for (int i = 1; i <= 4; ++i) {
        uint32_t octet = std::stoul(match[i].str());
        if (octet > 255)
          return 0; // Invalid IP
        ipInt = (ipInt << 8) | octet;
      }
      return ipInt;
    } catch (const std::exception &) {
      return 0; // Conversion error
    }
  }
  return 0; // Invalid format
}
pni_log::PrintHandler print(std::cout);
namespace cmdline {
std::string algorithmType = "socket";
std::string rawdataInputPath = "";
std::vector<std::string> addressExpressions;
int scanDuration = 60;
int maxPacketSize = 1024;
int minPacketSize = 1024;
int memoryUsageInGiB = 4;
bool forceRepalce = false;
std::string log_level;
bool skipStorage = false;
bool noStop = false;
uint16_t reservedStorageGiB{20};
bool silent = false;
int dpdkCopyThreadNum = 8;
int dpdkRxRings = 1;
std::vector<std::string> dpdkBindIPs = {};
int tickTime = 1000;
std::string speedSampleFile = "";
uint16_t rte_mbuf_double_pointer_size_multiply{32};
uint16_t rte_mbuf_double_pointer_num_pultiply{2};
} // namespace cmdline
template <typename T>
concept AcquisitionAlgorithm = std::is_same_v<T, socket::Socket>
#if PNI_STANDARD_CONFIG_ENABLE_DPDK
                               || std::is_same_v<T, dpdk::DPDK>
#endif
    ;

static AcquisitionInfo param;
#if PNI_STANDARD_CONFIG_ENABLE_DPDK
static dpdk::DPDKParam dpdk_info;
#endif
static std::unique_ptr<openpni::io::RawFileOutput> output;
std::unique_ptr<std::ofstream> fileSampling;
void prepare() {
  print.print().level(INFO).say("Preparing acquisition params.");

  param.storageUnitSize = cmdline::maxPacketSize;
  param.maxBufferSize = 1024ull * 1024ull * 1024ull * cmdline::memoryUsageInGiB;
  param.timeSwitchBuffer_ms = std::max<int>(cmdline::tickTime, 10);
  param.totalChannelNum = cmdline::addressExpressions.size();
  print.print().level(DEBUG).say(std::format("Filtering UDP packets whose length is outside range [{},{}]",
                                             cmdline::minPacketSize, param.storageUnitSize));
  print.print().level(DEBUG).say(std::format("Memory usage = {} GiB.", cmdline::memoryUsageInGiB));

  bool resolveAddressSuccess = true;
  cmdline::addressExpressions.erase(
      std::remove(cmdline::addressExpressions.begin(), cmdline::addressExpressions.end(), std::string()),
      cmdline::addressExpressions.end());

  if (!cmdline::addressExpressions.size()) {
    print.print().level(CRITICAL).say("No address to resolve.");
    exit(1);
  }
  print.print().level(DEBUG).say(std::format("There are {} addresses to resolve.", cmdline::addressExpressions.size()));
  for (int i = 0; i < cmdline::addressExpressions.size(); i++) {
    const auto &originFormat = cmdline::addressExpressions[i];
    AcquisitionInfo::ChannelSetting addressSetting;
    addressSetting.channelIndex = i;
    try {
      std::regex regexMatchIP(
          R"(^(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)==(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)$)");
      std::smatch matchIP;
      if (std::regex_match(originFormat, matchIP, regexMatchIP)) {
        addressSetting.ipSource = ipStr2ipInt(matchIP[1].str());
        addressSetting.portSource = std::stoi(matchIP[2].str());
        addressSetting.ipDestination = ipStr2ipInt(matchIP[3].str());
        addressSetting.portDestination = std::stoi(matchIP[4].str());
      } else {
        print.print().level(ERROR).say(std::format("Bad address format: {}", originFormat));
        resolveAddressSuccess = false;
      }
    } catch (const std::exception &e) {
      print.print().level(ERROR).say(std::format("Bad address format: {}", originFormat));
      resolveAddressSuccess = false;
    }
    addressSetting.quickFilter = [minPacketSize = cmdline::minPacketSize, maxPacketSize = cmdline::maxPacketSize](
                                     uint8_t *__udpDatagram, uint16_t __udpLength, uint32_t __ipSource,
                                     uint16_t __portSource) noexcept -> bool {
      if (__udpLength < minPacketSize || __udpLength > maxPacketSize)
        return false; // Filter out packets that are too small or too large
      return true;    // Accept the packet
    };

    param.channelSettings.push_back(addressSetting);
  }
  if (!resolveAddressSuccess) {
    print.print().level(CRITICAL).say("Resolve address failed.");
    exit(1);
  }

  if (cmdline::reservedStorageGiB == 0) {
    print.print().level(CRITICAL).say(
        "It is very dangerous to reserve no space for saving rawdata, thus the proccess must shutdown.");
    exit(1);
  }

  if (cmdline::skipStorage) {
    print.print().level(INFO).say("Skip storage.");
  } else {
    if (cmdline::rawdataInputPath.empty()) {
      print.print().level(CRITICAL).say("Must specify the output file path.");
      exit(1);
    }
    if (fs::exists(cmdline::rawdataInputPath) && !cmdline::forceRepalce) {
      print.print().level(CRITICAL).say(
          "The output file already exists. If you want to replace it anyway, try param --force-replace.");
      exit(1);
    }

    print.print().level(INFO).say(std::format("Opening file {}...", fs::absolute(cmdline::rawdataInputPath).string()));
    output = std::make_unique<openpni::io::RawFileOutput>();
    output->setReservedBytes(GIB * cmdline::reservedStorageGiB);
    try {
      if (fs::path(cmdline::rawdataInputPath).parent_path() != "")
        fs::create_directories(fs::path(cmdline::rawdataInputPath).parent_path());
      output->setChannelNum(param.totalChannelNum);
      output->open(cmdline::rawdataInputPath);
    } catch (const std::exception &e) {
      print.print().level(CRITICAL).say(e.what());
      print.print().level(CRITICAL).say(std::format("Cannot open file at {}", cmdline::rawdataInputPath));
      exit(1);
    }
  }
  if (!cmdline::speedSampleFile.empty()) {
    fileSampling = std::make_unique<std::ofstream>(cmdline::speedSampleFile);
    if (!fileSampling->is_open()) {
      print.print().level(CRITICAL).say(std::format("Cannot open file {}", cmdline::speedSampleFile));
      exit(1);
    }
  }

#if PNI_STANDARD_CONFIG_ENABLE_DPDK
  dpdk_info.copyThreadNum = cmdline::dpdkCopyThreadNum;
  dpdk_info.etherIpBind = cmdline::dpdkBindIPs;
  dpdk_info.rxThreadNumForEachPort = cmdline::dpdkRxRings;
  dpdk_info.rte_mbuf_double_pointer_size_multiply = cmdline::rte_mbuf_double_pointer_size_multiply;
  dpdk_info.rte_mbuf_double_pointer_num_pultiply = cmdline::rte_mbuf_double_pointer_num_pultiply;
  if (cmdline::algorithmType == "dpdk") {
    print.print().level(INFO).say("Initializing DPDK environment...");
    try {
      dpdk::InitDPDK(dpdk_info, [](const std::string &__message) { print.print().level(INFO).say(__message); });
    } catch (const std::exception &e) {
      print.print().level(CRITICAL).say("Some error has occurred when initializing dpdk.");
      exit(1);
    }
  }
#else
  if (cmdline::algorithmType == "dpdk") {
    print.print().level(CRITICAL).say("DPDK algorithm is not supported.");
    exit(1);
  }
#endif
}
template <typename T>
class MinMaxAvrCounter {
  const uint64_t m_timeVolume;
  std::vector<std::pair<uint64_t, T>> m_vecValue;

public:
  MinMaxAvrCounter(
      const uint64_t timeVolume)
      : m_timeVolume(timeVolume) {}
  void append(
      T value) {
    const uint64_t now = time_point_cast<milliseconds>(steady_clock::now()).time_since_epoch().count();
    m_vecValue.emplace_back(now, value);
    const auto iter = std::remove_if(m_vecValue.begin(), m_vecValue.end(), [this, now](const auto &item) {
      const auto &[time, value] = item;
      return now - time > m_timeVolume;
    });
    m_vecValue.erase(iter, m_vecValue.end());
  }
  std::tuple<T, T, T> operator()() {
    if (!m_vecValue.size())
      return {T(0), T(0), T(0)};

    T min{m_vecValue.begin()->second}, max{m_vecValue.begin()->second}, avr{0};
    for (const auto &[time, item] : m_vecValue) {
      if (item > max)
        max = item;
      if (item < min)
        min = item;
      avr += item;
    }
    avr /= m_vecValue.size();
    return {min, max, avr};
  }
};
bool bufferWarning(
    socket::Status s) {
  return s.used / double(s.volume) >= 0.5;
}
std::string printBuffer(
    socket::Status s) {
  return std::format("{}/{},u'{}'", s.used, s.volume, s.unknown);
}
#if PNI_STANDARD_CONFIG_ENABLE_DPDK
rte_eth_stats dpdk_status_addup() {
  rte_eth_stats result{0};
  for (int i = 0; i < dpdk::rte_port_num; i++) {
    result.ierrors += dpdk::dpdk_status[i].rte_stats.ierrors;
    result.imissed += dpdk::dpdk_status[i].rte_stats.imissed;
    result.rx_nombuf += dpdk::dpdk_status[i].rte_stats.rx_nombuf;
  }
  return result;
}
std::string printBuffer(
    dpdk::Status s) {
  const auto status = dpdk_status_addup();
  return std::format("{}/{},e'{}',m'{}',n'{}',u'{}'", s.used, s.volume, status.ierrors, status.imissed,
                     status.rx_nombuf, s.unknown);
}
std::string printBufferEachPort(
    socket::Status s) {
  return std::format("{}/{},u'{}'", s.used, s.volume, s.unknown);
}

std::string printBufferEachPort(
    dpdk::Status s) {
  std::string result = std::format("{}/{}", s.used, s.volume);
  result += std::format(",u'{}'", s.unknown);
  return result;
}
void printSecondaryInformation() {
  for (int i = 0; i < dpdk::rte_port_num; i++) {
    print.print().level(DEBUG).say(
        std::format("p{}: i\'{},o\'{},m\'{}%,ie\'{},oe\'{},n\'{},f\'{},v\'{}", i,
                    dpdk::dpdk_status[i].rte_stats.ipackets, dpdk::dpdk_status[i].rte_stats.opackets,
                    dpdk::dpdk_status[i].rte_stats.imissed, dpdk::dpdk_status[i].rte_stats.ierrors,
                    dpdk::dpdk_status[i].rte_stats.oerrors, dpdk::dpdk_status[i].rte_stats.rx_nombuf,
                    dpdk::dpdk_status[i].dpdkBufferFree, dpdk::dpdk_status[i].dpdkBufferVolume));
  }
}
void printSecondaryInformation(
    socket::Status s) {
  return;
}
bool dpdkBufferWarning() {
  for (int i = 0; i < dpdk::rte_port_num; i++)
    if (dpdk::dpdk_status[i].dpdkBufferFree / double(dpdk::dpdk_status[i].dpdkBufferVolume) <= 0.5)
      return true;
  return false;
}
bool bufferWarning(
    dpdk::Status s) {
  return s.used / double(s.volume) >= 0.5 || dpdkBufferWarning();
}
#endif
template <AcquisitionAlgorithm AA>
void run() {
  print.print().level(INFO).say("Starting acquisition of type " + cmdline::algorithmType);
  std::mutex mutex;
  auto k = AA(param, [&mutex](const std::string &message) {
    std::lock_guard ll(mutex);
    print.print().level(INFO).say(message);
  });
  try {
    if (!k.start()) {
      print.print().level(CRITICAL).say("Failed to start acquisition.");
      exit(1);
    }
  } catch (const std::exception &e) {
    print.print().level(CRITICAL).say(e.what());
    exit(1);
  }

  if (cmdline::noStop)
    print.print().level(INFO).say("Run in no-stop mode.");

  uint64_t acquisitionTimeSecond = cmdline::scanDuration;
  const auto startTime = steady_clock::now();
  constexpr double debugEveryMicro = 1000000;

  uint64_t totalReadCount{0};

  auto endTime = steady_clock::now();
  bool stopped{false};
  bool signalStopNow{false};
  uint64_t totalBytes{0};
  const auto funcStop = [&] {
    if (stopped)
      return;
    print.print().level(INFO).say("Sent stop signal.");
    k.stop();
    endTime = steady_clock::now();
    stopped = true;
  };

  std::thread([&] {
    while (true) {
      std::string input;
      std::cin >> input;
      if (input == "q" || input == "Q") {
        signalStopNow = true;
        print.print().level(INFO).say("Input stop signal.");
        return;
      } else {
        print.print().level(WARNING).say("Unknown command.");
      }
    }
  }).detach();
  auto threadWatch = std::jthread([&] {
    uint64_t logTime{0};
    MinMaxAvrCounter<double> minMaxAvrCounter(1500);
    while (!stopped) {
      const auto nowTime = steady_clock::now();
      const auto nowMilli = duration_cast<microseconds>(nowTime - startTime).count();
      if (signalStopNow)
        funcStop();

      if (nowMilli > acquisitionTimeSecond * 1000000 && !cmdline::noStop)
        funcStop();

      const auto status = k.status();
      static uint64_t lastRxBytes{0};
      static uint64_t lastMilli{0};
      const auto speed = (status.totalRxBytes - lastRxBytes) / 1000. / 1000. /
                         double(std::max<uint64_t>(nowMilli - lastMilli, 1)) * 1000000.;
      lastRxBytes = status.totalRxBytes;
      lastMilli = nowMilli;
      minMaxAvrCounter.append(speed);
      totalBytes = status.totalRxBytes;

      if (fileSampling)
        *fileSampling << speed << "  \n";

      if (nowMilli - logTime * debugEveryMicro > debugEveryMicro && !stopped) {
        logTime = nowMilli / debugEveryMicro;
        const auto [min, max, avr] = minMaxAvrCounter();
        if constexpr (std::is_same_v<AA, socket::Socket>) {
          print.print()
              .level(DEBUG)
              .block(!bufferWarning(status))
              .warning()
              .say(std::format("R\'{} b\'{} min: {:.4f}MB/s max: {:.4f}MB/s avr: {:.4f}MB/s", status.totalRxPackets,
                               printBuffer(status), min, max, avr))
              .reset();
        }

#if PNI_STANDARD_CONFIG_ENABLE_DPDK
        else if constexpr (std::is_same_v<AA, dpdk::DPDK>) {
          print.print()
              .level(DEBUG)
              .block(!bufferWarning(status))
              .warning()
              .say(std::format("R\'{} b\'{} min: {:.4f}MB/s max: {:.4f}MB/s avr: {:.4f}MB/s", status.totalRxPackets,
                               printBuffer(status), min, max, avr))
              .reset();
          printSecondaryInformation();
        }
#endif
        std::this_thread::sleep_for(microseconds(5000));
      }
    }
  });

  auto handler = openpni::process::read_from_acquisition(std::ref(k));

  while (const auto data = handler.get()) {
    totalReadCount += data->count;
    const auto speedMpps = data->count * 1000. / data->duration_ms / 1024 / 1024;
    // _LOG_(DEBUG) << "Read RX=" << data->count << " at=" << clock(data->clock) << "ms duration=" <<
    // clock(data->duration) << "ms " << (cmdline::skipStorage ? " skipped" : "");
    if (output) {
      if (!output->appendSegment(data.value())) {
        print.print().level(ERROR).say(
            std::format("There is no enough space in file system, available = {}GiB",
                        std::to_string(fs::space(cmdline::rawdataInputPath).available / GIB)));
        if (cmdline::noStop)
          funcStop();
      }
    }
  }

  print.print().level(INFO).say(std::format("Total RX: {}", totalReadCount));
  const auto speedMpps =
      totalReadCount * 1000. / duration_cast<milliseconds>(endTime - startTime).count() / 1024 / 1024;
  print.print().level(INFO).say(
      std::format("Average Speed: {:.4f}Mpps {:.4f}MB/s", speedMpps,
                  totalBytes / 1024. / 1024. / duration_cast<milliseconds>(endTime - startTime).count() * 1000.));
  return;
}
cmd::ParseResult result;
int main(
    int argc, char *argv[]) {
  cmd::Options options("pni-aqst", "The PnI cmd-program for data acquisition.");

  options.add_options("Main")("h,help", "Print usage");
  options.add_options("Main")("algo", "The algorithm for data acquisition, supported: socket(recommended),dpdk,dpu",
                              cmd::value(cmdline::algorithmType), "TYPE");
  options.add_options("Main")("o,out", "The output path for data", cmd::value(cmdline::rawdataInputPath), "FILE");
  options.add_options("Main")(
      "address", R"(The formatted ipv4 addresses. Format: source==destination [0.0.0.0:8000==255.255.255.255:65535])",
      cmd::value<std::vector<std::string>>(cmdline::addressExpressions), "[ADDR]");
  options.add_options("Main")("d,duration", "The time duration (unit:s) of acquisition.",
                              cmd::value<int>(cmdline::scanDuration), "TIME");
  options.add_options("Main")("m,min", "The minimun size of UDP datagram to accept.",
                              cmd::value<int>(cmdline::minPacketSize)->default_value("1024"), "SIZE");
  options.add_options("Main")("M,max", "The maximun size of UDP datagram to accept.",
                              cmd::value<int>(cmdline::maxPacketSize)->default_value("1024"), "SIZE");
  options.add_options("Main")("t,tick", "The time intercept for data read-out.",
                              cmd::value(cmdline::tickTime)->default_value("1000"), "ms");

  options.add_options("Other")("force-replace", "If the target output path exists, replace it with new file.",
                               cmd::value<bool>(cmdline::forceRepalce));
  options.add_options("Other")("mem-usage", "The memory usage (GiB).",
                               cmd::value<int>(cmdline::memoryUsageInGiB)->default_value("4"), "SIZE");
  options.add_options("Other")("skip-storage", "If set, it will NOT actually save rawdata to disk.",
                               cmd::value<bool>(cmdline::skipStorage));
  options.add_options("Other")("no-stop", "Keep acquiring data, until the storage is full.",
                               cmd::value<bool>(cmdline::noStop));
  options.add_options("Other")(
      "storage-reserved",
      "To protect the file system, the process will refuse to actually write to disk if there is no enough space.",
      cmd::value(cmdline::reservedStorageGiB)->default_value("20"), "GiB");

  options.add_options("DPDK")(
      "dpdk-bind-ip",
      "When running in dpdk, it is neccessary to bind ip to the physical port. Each physical port needs an IP.",
      cmd::value(cmdline::dpdkBindIPs), "IP");
  options.add_options("DPDK")("dpdk-copy-cores", "The thread num for dpdk memory copy.",
                              cmd::value(cmdline::dpdkCopyThreadNum));
  options.add_options("DPDK")("dpdk-rx-rings", "Specify how many Rx rings for each port.",
                              cmd::value(cmdline::dpdkRxRings)->default_value("1"), "NUM");
  options.add_options("DPDK")("dpdk-buffer-ppsize-multiply",
                              "Specify the multiply ratio for the size of each internal buffer item.",
                              cmd::value(cmdline::rte_mbuf_double_pointer_size_multiply)->default_value("32"), "NUM");
  options.add_options("DPDK")("dpdk-buffer-ppnum-multiply",
                              "Specify the multiply ratio for the number of internal buffer items.",
                              cmd::value(cmdline::rte_mbuf_double_pointer_num_pultiply)->default_value("2"), "NUM");

  options.add_options("Information")(
      "log-level", "Specify the log-level of the process. Value={DEBUG,INFO,WARNING,ERROR,CRITICAL}, Default=INFO",
      cmd::value<std::string>(cmdline::log_level)->default_value("INFO"));
  options.add_options("Information")("silent", "Do NOT print anything in the terminal.",
                                     cmd::value<bool>(cmdline::silent));
  options.add_options("Information")("speed-sample", "Specify the output path for speed sampling.",
                                     cmd::value(cmdline::speedSampleFile), "PATH");

  try {
    result = options.parse(argc, argv);
  } catch (const std::exception &e) {
    if (!cmdline::silent) {
      std::cerr << "pni-aqst: Failed to parse command line options: " << e.what() << ".\n\n";
      std::cout << options.help() << std::endl;
    }
    exit(1);
  }
  if (result.unmatched().size()) {
    if (!cmdline::silent) {
      std::cout << "There are some params not matched: ";
      for (int i = 0; i < result.unmatched().size(); i++) {
        if (i != 0)
          std::cout << ",";
        std::cout << result.unmatched()[i];
      }
      std::cout << std::endl << std::endl;
      std::cout << options.help() << std::endl;
    }
    exit(0);
  }
  if (result.count("help") || !result.arguments().size()) {
    if (!cmdline::silent)
      std::cout << options.help() << std::endl;
    exit(0);
  }

  if (cmdline::silent)
    cmdline::log_level = "SILENT";
  if (!pni_log::mapLogLevels.contains(cmdline::log_level)) {
    const auto ll = cmdline::log_level;
    cmdline::log_level = "INFO";
    print.print().level(WARNING).say("Unknown log level = " + ll + ", reset to \"INFO\".");
  }
  //   pni_log::minLogLevel = pni_log::mapLogLevels.at(cmdline::log_level);

  prepare();

  if (cmdline::algorithmType == "socket")
    run<socket::Socket>();
#if PNI_STANDARD_CONFIG_ENABLE_DPDK
  else if (cmdline::algorithmType == "dpdk")
    run<dpdk::DPDK>();
#endif
  else {
    print.print().level(CRITICAL).say("Unknown algorithm = " + cmdline::algorithmType);
    exit(1);
  }

  return 0;
}

static auto __timeReset = []() -> int {
  pni_log::time(false);
  return 1;
}();