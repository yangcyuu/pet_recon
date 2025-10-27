#include <chrono>
#include <cxxopts.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <ranges>
#include <regex>
#include <set>
#include <thread>

#include "../print.hpp"
#include "include/io/IO.hpp"
#include "include/process/Acquisition.hpp"
namespace cmd = cxxopts;
namespace fs = std::filesystem;
using namespace openpni;
using namespace std::chrono;
#define _LOG_(LEVEL, CONTENT) pni_log::log(pni_log::LogLevel::LEVEL, ((std::stringstream() << CONTENT).str()))
#define GIB (1024ull * 1024ull * 1024ull)
namespace cmdline {
std::string rawdataInputPath = "";
std::string outputDir = ".";
std::string outputName = "";
bool forceRepalce = false;
bool operation_show = false;
bool operation_toHex = false;
bool operation_clipSegment = false;
bool operation_seperateChannel = false;
bool operation_showChannel = false;
} // namespace cmdline
void show(
    openpni::io::RawFileInput &in) {
  std::vector<bool> segmentAssert(in.segmentNum(), true);

  std::cout << "File " << cmdline::rawdataInputPath << " information:" << std::endl;
  const auto header = in.header();
  std::cout << "TypeName = " << header.fileTypeName << std::endl;
  std::cout << "BDM type = [";
  for (uint16_t i = 0; i < header.channelNum; i++) {
    std::cout << in.typeNameOfChannel(i);
    if (i != header.channelNum - 1)
      std::cout << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << "ChannelNum = " << header.channelNum << std::endl;
  std::cout << "SegmentNum = " << header.segmentNum << std::endl;
  for (int i = 0; i < in.segmentNum(); i++) {
    const auto segmentHeader = in.segmentHeader(i);
    std::cout << "Segment[" << i << "]: ";
    std::cout << "start=" << segmentHeader.clock << "ms, ";
    std::cout << "duration=" << segmentHeader.duration << "ms, ";
    std::cout << "count=" << segmentHeader.count << " ";
    std::cout << std::endl;
    if (cmdline::operation_showChannel) {
      const auto segment = in.readSegment(i, i + 1);
      const auto segmentHeader = in.segmentHeader(i);
      std::map<uint16_t, uint32_t> channelCount;
      for (uint32_t j = 0; j < segmentHeader.count; j++) {
        if (channelCount.contains(segment.channel[j]))
          channelCount[segment.channel[j]]++;
        else
          channelCount[segment.channel[j]] = 1;
      }
      for (const auto &pair : channelCount) {
        std::cout << "Channel " << pair.first << ": " << pair.second << " packets\n";
      }

      if (segmentHeader.count) {
        do {
          if (!!segment.offset[0]) {
            std::cout << "Assert = failed: offset[0] != 0" << std::endl;
            segmentAssert[i] = false;
            break;
          }
          for (uint32_t j = 1; j < segmentHeader.count; j++) {
            if (segment.offset[j - 1] + segment.length[j - 1] != segment.offset[j]) {
              std::cout << "Assert = failed: offset[j-1] + length[j-1] != offset[j]" << std::endl;
              segmentAssert[i] = false;
              break;
            }
          }
          std::cout << "Assert = true" << std::endl;

        } while (0);
      } else {
        std::cout << "Assert = true" << std::endl;
      }
    }
  }

  if (cmdline::operation_showChannel) {
    std::cout << "File assert: \n";
    for (int i = 0; i < in.segmentNum(); i++) {
      if (segmentAssert[i])
        std::cout << "Segment[" << i << "] =" << termcolor::green << " true " << termcolor::reset << std::endl;
      else
        std::cout << "Segment[" << i << "] =" << termcolor::red << " false " << termcolor::reset << std::endl;
    }
  }
}
struct SegmentInfo {
  uint16_t channelIndex;
  uint32_t segmentIndex;
};
std::vector<std::pair<uint32_t, std::vector<uint16_t>>> rearrangeSegmentInfo(
    const std::vector<SegmentInfo> &orderedPairs) {
  std::vector<std::pair<uint32_t, std::vector<uint16_t>>> result;
  for (const auto &pair : orderedPairs) {
    auto iter =
        std::find_if(result.begin(), result.end(), [&pair](const std::pair<uint32_t, std::vector<uint16_t>> &p) {
          return p.first == pair.segmentIndex;
        });
    if (iter != result.end())
      iter->second.push_back(pair.channelIndex);
    else
      result.push_back({pair.segmentIndex, {pair.channelIndex}});
  }
  return result;
}

struct FileOperation // 文件操作结构体，一个结构体代表一个文件
{
  std::vector<SegmentInfo> orderedPairs; // 按照通道和分段的顺序排列
  std::string outputPath;                // 输出文件路径
};
template <typename List>
bool listContains(
    const List &list, const typename List::value_type &value) {
  return std::find(list.begin(), list.end(), value) != list.end();
}
void appendHexToStream(
    std::ostream &out, const uint8_t *data, const uint16_t *length, const uint64_t *offset, const uint16_t *channel,
    const uint32_t count, std::vector<uint16_t> channelsWanted) {
  std::stringstream buffer;
  for (uint32_t i = 0; i < count; i++) {
    if (!listContains(channelsWanted, channel[i]))
      continue; // 如果当前通道不在需要的通道列表中，则跳过
    const uint8_t *begin = data + offset[i];
    const uint8_t *end = begin + length[i];
    for (const uint8_t *p = begin; p < end; p++) {
      buffer << "0123456789abcdef"[(*p >> 4) & 0x0F];
      buffer << "0123456789abcdef"[(*p >> 0) & 0x0F];
    }
    buffer << '\n';
  }
  out << buffer.str();
}
io::rawdata::RawdataSegment generateClippedSegment(
    const io::rawdata::RawdataSegment &input, uint64_t inputCount, std::map<uint16_t, uint16_t> &outputChannelMapper,
    uint64_t &outputCount) {
  uint64_t resultCount = 0;
  uint64_t resultTotalBytes = 0;

  for (uint64_t i = 0; i < inputCount; i++) {
    if (outputChannelMapper.contains(input.channel[i])) {
      resultCount++;
      resultTotalBytes += input.length[i];
    }
  }

  io::rawdata::RawdataSegment result;
  result.data = std::make_unique<uint8_t[]>(resultTotalBytes);
  result.length = std::make_unique<uint16_t[]>(resultCount);
  result.offset = std::make_unique<uint64_t[]>(resultCount);
  result.channel = std::make_unique<uint16_t[]>(resultCount);
  outputCount = resultCount;

  uint64_t currentOffset = 0;
  uint64_t currentIndex = 0;
  for (uint64_t i = 0; i < inputCount; i++) {
    if (outputChannelMapper.contains(input.channel[i])) {
      result.length[currentIndex] = input.length[i];
      result.offset[currentIndex] = currentOffset;
      result.channel[currentIndex] = outputChannelMapper[input.channel[i]];
      ::memcpy(result.data.get() + currentOffset, input.data.get() + input.offset[i], input.length[i]);
      currentOffset += input.length[i];
      currentIndex++;
    }
  }

  return result;
};
std::vector<FileOperation> generateFileOperations(const uint16_t channelNum, const uint32_t segmentNum);
void Output2FileAsHex(const std::string &outputPath,
                      const std::vector<std::pair<uint32_t, std::vector<uint16_t>>> &rearrangedPairs,
                      int &lastSegmentIndex, openpni::io::rawdata::RawdataSegment &segment,
                      openpni::io::RawFileInput &in);
void fileOperation(
    openpni::io::RawFileInput &in) {
  if (cmdline::outputName.empty()) {
    std::cerr << "Output file name is empty, please specify it with -o option." << std::endl;
    exit(1);
  }
  if (cmdline::outputDir.empty()) {
    std::cerr << "Output directory is empty, please specify it with -O option." << std::endl;
    exit(1);
  }

  if (!fs::exists(cmdline::outputDir))
    fs::create_directories(cmdline::outputDir);
  if (!fs::is_directory(cmdline::outputDir)) {
    std::cerr << "Output directory " << cmdline::outputDir << " is not a directory." << std::endl;
    exit(1);
  }

  if (!cmdline::operation_toHex && !cmdline::operation_clipSegment && !cmdline::operation_seperateChannel) {
    std::cerr << "No operation to do." << std::endl;
    exit(1);
  }

  std::vector<FileOperation> fileOperations;
  const auto channelNum = in.header().channelNum;
  const auto segmentNum = in.segmentNum();
  fileOperations = generateFileOperations(channelNum, segmentNum);

  if (fileOperations.empty()) {
    std::cerr << "No file is generated." << std::endl;
  }

  io::rawdata::RawdataSegment segment;
  int lastSegmentIndex = -1;
  for (const auto &fileOperation : fileOperations) {
    const auto rearrangedPairs = rearrangeSegmentInfo(fileOperation.orderedPairs);
    const auto &outputPath = fileOperation.outputPath;
    if (cmdline::operation_toHex) {
      Output2FileAsHex(outputPath, rearrangedPairs, lastSegmentIndex, segment, in);
    } else {
      io::RawFileOutput outFile;
      try {
        outFile.open(outputPath);
      } catch (const std::exception &e) {
        std::cerr << "Failed to open output file " << outputPath << " for writing: " << e.what() << std::endl;
        exit(1);
      }
      if (cmdline::forceRepalce && fs::exists(outputPath)) {
        fs::remove(outputPath);
      }

      std::map<uint16_t, uint16_t> outputChannelMapper;
      for (const auto &[segmentIndex, channels] : rearrangedPairs)
        for (auto channel : channels) {
          if (outputChannelMapper.contains(channel))
            continue; // 如果通道已经存在，则跳过
          const auto mapperSize = outputChannelMapper.size();
          outputChannelMapper[channel] = static_cast<uint16_t>(mapperSize);
        }

      outFile.setChannelNum(outputChannelMapper.size());
      for (const auto &[segmentIndex, channels] : rearrangedPairs) {
        if (segmentIndex != lastSegmentIndex) {
          segment = in.readSegment(segmentIndex, segmentIndex + 1);
          lastSegmentIndex = segmentIndex;
        }
        const auto segHeader = in.segmentHeader(segmentIndex);
        uint64_t outputCount = 0;
        io::rawdata::RawdataSegment clippedSegment =
            generateClippedSegment(segment, segHeader.count, outputChannelMapper, outputCount);
        openpni::process::RawDataView viewOfClippedSegment;
        viewOfClippedSegment.data = clippedSegment.data.get();
        viewOfClippedSegment.length = clippedSegment.length.get();
        viewOfClippedSegment.offset = clippedSegment.offset.get();
        viewOfClippedSegment.channel = clippedSegment.channel.get();
        viewOfClippedSegment.count = outputCount;
        viewOfClippedSegment.clock_ms = segHeader.clock;
        viewOfClippedSegment.duration_ms = segHeader.duration;
        viewOfClippedSegment.channelNum = outputChannelMapper.size();
        outFile.appendSegment(viewOfClippedSegment);
      }
    }
  }
}
void Output2FileAsHex(
    const std::string &outputPath, const std::vector<std::pair<uint32_t, std::vector<uint16_t>>> &rearrangedPairs,
    int &lastSegmentIndex, openpni::io::rawdata::RawdataSegment &segment, openpni::io::RawFileInput &in) {
  if (!cmdline::forceRepalce && fs::exists(outputPath)) {
    std::cerr << "Output file " << outputPath << " already exists, please use --force-replace to replace it. Stop."
              << std::endl;
    exit(1);
  }
  std::ofstream outFile(outputPath, std::ios::out);
  if (!outFile.is_open()) {
    std::cerr << "Failed to open output file " << outputPath << " for writing." << std::endl;
    exit(1);
  }
  for (auto [segmentIndex, channels] : rearrangedPairs) {
    if (segmentIndex != lastSegmentIndex) {
      segment = in.readSegment(segmentIndex, segmentIndex + 1);
      lastSegmentIndex = segmentIndex;
    }
    const auto segHeader = in.segmentHeader(segmentIndex);
    appendHexToStream(outFile, segment.data.get(), segment.length.get(), segment.offset.get(), segment.channel.get(),
                      segHeader.count, channels);
  }
}
std::vector<FileOperation> generateFileOperations(
    const uint16_t channelNum, const uint32_t segmentNum) {
  std::vector<FileOperation> fileOperations;
  if (cmdline::operation_clipSegment && cmdline::operation_seperateChannel) {
    for (int i = 0; i < channelNum; i++)
      for (int j = 0; j < segmentNum; j++) {
        FileOperation op;
        op.orderedPairs.push_back({static_cast<uint16_t>(i), static_cast<uint32_t>(j)});
        std::string outputPath = cmdline::outputDir + "/" + cmdline::outputName;
        outputPath = std::regex_replace(outputPath, std::regex("\\{channel\\}"), std::to_string(i));
        outputPath = std::regex_replace(outputPath, std::regex("\\{segment\\}"), std::to_string(j));
        op.outputPath = outputPath;
        fileOperations.push_back(op);
      }
  } else if (cmdline::operation_clipSegment && !cmdline::operation_seperateChannel) {
    for (int j = 0; j < segmentNum; j++) {
      FileOperation op;
      for (int i = 0; i < channelNum; i++)
        op.orderedPairs.push_back({static_cast<uint16_t>(i), static_cast<uint32_t>(j)});
      std::string outputPath = cmdline::outputDir + "/" + cmdline::outputName;
      outputPath = std::regex_replace(outputPath, std::regex("\\{segment\\}"), std::to_string(j));
      outputPath = std::regex_replace(outputPath, std::regex("\\{channel\\}"), "all-channles");
      op.outputPath = outputPath;
      fileOperations.push_back(op);
    }
  } else if (!cmdline::operation_clipSegment && cmdline::operation_seperateChannel) {
    for (int i = 0; i < channelNum; i++) {
      FileOperation op;
      for (int j = 0; j < segmentNum; j++)
        op.orderedPairs.push_back({static_cast<uint16_t>(i), static_cast<uint32_t>(j)});
      std::string outputPath = cmdline::outputDir + "/" + cmdline::outputName;
      outputPath = std::regex_replace(outputPath, std::regex("\\{channel\\}"), std::to_string(i));
      outputPath = std::regex_replace(outputPath, std::regex("\\{segment\\}"), "all-segments");
      op.outputPath = outputPath;
      fileOperations.push_back(op);
    }
  } else {
    FileOperation op;
    for (int i = 0; i < channelNum; i++)
      for (int j = 0; j < segmentNum; j++)
        op.orderedPairs.push_back({static_cast<uint16_t>(i), static_cast<uint32_t>(j)});
    std::string outputPath = cmdline::outputDir + "/" + cmdline::outputName;
    outputPath = std::regex_replace(outputPath, std::regex("\\{channel\\}"), "all-channels");
    outputPath = std::regex_replace(outputPath, std::regex("\\{segment\\}"), "all-segments");
    outputPath = outputPath;
    fileOperations.push_back(op);
  }
  return fileOperations;
}
void run() {
  io::RawFileInput input;
  try {
    input.open(cmdline::rawdataInputPath);
  } catch (const std::exception &e) {
    std::cerr << "Failed to read file " << cmdline::rawdataInputPath << " because an error has occurred: " << e.what()
              << '\n';
    exit(1);
  }

  if (cmdline::operation_show)
    show(input);
  else if (cmdline::operation_toHex || cmdline::operation_clipSegment || cmdline::operation_seperateChannel)
    fileOperation(input);
  else {
    std::cerr << "Nothing to do.\n";
    exit(1);
  }
}
cmd::ParseResult result;
int main(
    int argc, char *argv[]) {
  cmd::Options options("pni-rawdata", "The PnI tools for rawdata file operations.");

  options.add_options("Main")("h,help", "Print usage");
  options.add_options("Main")("i,in", "The input rawdata file path.",
                              cmd::value<std::string>(cmdline::rawdataInputPath), "FILE");
  options.add_options("Main")("O,out-dir", "The output rawdata file directory. Default current directory.",
                              cmd::value<std::string>(cmdline::outputDir), "DIR");
  options.add_options("Main")("o,out-name", "The output rawdata file name. Regex format supports: {channel}, {segment}",
                              cmd::value<std::string>(cmdline::outputName), "REGEX");

  options.add_options("Operations")("info", "Output rawdata file info. Do NOT execute other file operations",
                                    cmd::value<bool>(cmdline::operation_show));
  options.add_options("Operations")("channel-info", "", cmd::value<bool>(cmdline::operation_showChannel));
  options.add_options("Operations")("to-hex", "Output rawdata in hex-string mode.",
                                    cmd::value<bool>(cmdline::operation_toHex));
  options.add_options("Operations")("clip-segment",
                                    "Output rawdata into several files, each of which contains one segment only.",
                                    cmd::value<bool>(cmdline::operation_clipSegment));
  options.add_options("Operations")("seperate-channel",
                                    "Output rawdata into several files, each of which contains one channel only.",
                                    cmd::value<bool>(cmdline::operation_seperateChannel));

  options.add_options("Other")("force-replace", "If the target output path exists, replace it with new file.",
                               cmd::value<bool>(cmdline::forceRepalce), "FILE");

  try {
    result = options.parse(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "pni-aqst: Failed to parse command line options: " << e.what() << ".\n\n";
    std::cout << options.help() << std::endl;
    exit(1);
  }
  if (result.unmatched().size()) {
    std::cout << "There are some params not matched: ";
    for (int i = 0; i < result.unmatched().size(); i++) {
      if (i != 0)
        std::cout << ",";
      std::cout << result.unmatched()[i];
    }
    std::cout << std::endl << std::endl;
    std::cout << options.help() << std::endl;
    exit(0);
  }
  if (result.count("help") || !result.arguments().size()) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  run();

  return 0;
}