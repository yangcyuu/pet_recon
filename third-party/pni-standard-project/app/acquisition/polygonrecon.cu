#include <chrono>
#include <cxxopts.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "include/detector/BDM2.hpp"
#include "include/example/PolygonRecon.cuh"

namespace cmd = cxxopts;
namespace fs = std::filesystem;
using namespace std::chrono;

namespace cmdline {
int subsetNum = 1;
int iterNum = 4;
int binCut = 0;
float hfwhm = 0.33f;

std::string imgSize = "320x320x400";
int imageWidth = 320;
int imageHeight = 320;
int imageDepth = 400;
// float sampleRatio = 1.0f;
std::string michdataInputPath = "";
int michdataOffset = 6;
std::string outputImagePath = ".";
std::string CorrAddPath = "";
std::string CorrMulPath = "";
} // namespace cmdline

void run() {
  if (cmdline::michdataInputPath.empty()) {
    std::cerr << "Input mich data file path is empty, please specify it with -i option." << std::endl;
    exit(1);
  }
  if (cmdline::outputImagePath.empty()) {
    std::cerr << "Output image file path is empty, please specify it with -o option." << std::endl;
    exit(1);
  }

  if (!fs::exists(cmdline::michdataInputPath)) {
    std::cerr << "Input mich data file " << cmdline::michdataInputPath << " does not exist." << std::endl;
    exit(1);
  }
  // check output directory
  if (fs::exists(cmdline::outputImagePath)) {
    std::cout << "Warning: Output file " << cmdline::outputImagePath << " already exists and will be overwritten."
              << std::endl;
  }

  auto e180 = openpni::example::E180();
  auto e180Builder = openpni::example::polygon::PolygonModelBuilder<openpni::device::bdm2::BDM2Runtime>(e180);
  auto e180System = e180Builder.build();

  std::vector<float> mich(e180System->michSize(), 0);
  openpni::example::readFile(cmdline::michdataInputPath, mich, cmdline::michdataOffset);
  std::vector<float> multi(e180System->michSize(), 1);
  openpni::example::readFile(cmdline::CorrMulPath, multi, 0);
  std::vector<float> add(e180System->michSize(), 0);
  openpni::example::readFile(cmdline::CorrAddPath, add, 0);

  auto param = openpni::example::OSEM_params();
  param.subsetNum = cmdline::subsetNum;
  param.iterNum = cmdline::iterNum;
  param.binCut = cmdline::binCut;
  param.hfwhm = cmdline::hfwhm;
  param.OSEMImgVoxelNum = {cmdline::imageWidth, cmdline::imageHeight, cmdline::imageDepth};
  // param.sampleRatio = cmdline::sampleRatio;

  auto imgGeometry =
      openpni::example::defaultImgGeometry(cmdline::imageWidth, cmdline::imageHeight, cmdline::imageDepth);
  std::vector<float> img(imgGeometry.totalVoxelNum(), 0);

  // doing osem reconstruction
  openpni::example::OSEM_CUDA(img.data(), mich.data(), param, *e180System, add.data(), multi.data());
  // openpni::example::OSEM_CUDA(img.data(), mich.data(), param, *e180System, nullptr, nullptr);

  // save image
  std::ofstream outFile(cmdline::outputImagePath, std::ios::binary);
  if (!outFile.is_open()) {
    std::cerr << "Failed to open output file: " << cmdline::outputImagePath << std::endl;
    exit(1);
  }

  outFile.write(reinterpret_cast<const char *>(img.data()), img.size() * sizeof(float));
  outFile.close();

  if (outFile.fail()) {
    std::cerr << "Failed to write to output file: " << cmdline::outputImagePath << std::endl;
    exit(1);
  }

  std::cout << "Reconstruction finished, output image saved to " << cmdline::outputImagePath << std::endl;
}

cmd::ParseResult result;
int main(
    int argc, char *argv[]) {
  cmd::Options options("pni-polygonrecon", "The PnI tools for polygon reconstruction.");

  options.add_options("Main")("h,help", "Print usage");
  options.add_options("Main")("i,in", "The input mich data file path.",
                              cmd::value<std::string>(cmdline::michdataInputPath), "FILE");
  options.add_options("Main")("o,out", "The output image file path. Default current directory.",
                              cmd::value<std::string>(cmdline::outputImagePath), "FILE");
  options.add_options("Main")("offset", "The offset in bytes for mich data file.",
                              cmd::value<int>(cmdline::michdataOffset)->default_value("6"), "INT");

  options.add_options("Reconstruction Parameters")("subset-num", "Number of subsets for OSEM reconstruction.",
                                                   cmd::value<int>(cmdline::subsetNum), "INT");
  options.add_options("Reconstruction Parameters")("iter-num", "Number of iterations for OSEM reconstruction.",
                                                   cmd::value<int>(cmdline::iterNum), "INT");
  options.add_options("Reconstruction Parameters")("bin-cut", "Binary cut parameter.", cmd::value<int>(cmdline::binCut),
                                                   "INT");
  options.add_options("Reconstruction Parameters")("hfwhm", "Half Full Width at Half Maximum.",
                                                   cmd::value<float>(cmdline::hfwhm), "FLOAT");

  options.add_options("Image Parameters")("img-size", "Image size in format WIDTHxHEIGHTxDEPTH.",
                                          cmd::value<std::string>(cmdline::imgSize), "STRING");
  //   options.add_options("Image Parameters")("sample-ratio", "Sample ratio for reconstruction.",
  //                                           cmd::value<float>(cmdline::sampleRatio), "FLOAT");

  options.add_options("Correction")("corr-add", "Additive correction file path.",
                                    cmd::value<std::string>(cmdline::CorrAddPath), "FILE");
  options.add_options("Correction")("corr-mul", "Multiplicative correction file path.",
                                    cmd::value<std::string>(cmdline::CorrMulPath), "FILE");

  try {
    result = options.parse(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "pni-polygonrecon: Failed to parse command line options: " << e.what() << ".\n\n";
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

  // 解析图像尺寸字符串
  if (result.count("img-size")) {
    std::regex sizeRegex(R"((\d+)x(\d+)x(\d+))");
    std::smatch matches;
    if (std::regex_match(cmdline::imgSize, matches, sizeRegex)) {
      cmdline::imageWidth = std::stoi(matches[1].str());
      cmdline::imageHeight = std::stoi(matches[2].str());
      cmdline::imageDepth = std::stoi(matches[3].str());
    } else {
      std::cerr << "Invalid image size format. Expected format: WIDTHxHEIGHTxDEPTH" << std::endl;
      exit(1);
    }
  }

  run();

  return 0;
}