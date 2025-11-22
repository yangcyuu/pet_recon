#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <stdio.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../basic/Image.hpp"
#include "../basic/Triangulation.hpp"
#include "../detector/Detectors.hpp"
#include "../example/PolygonalSystem.hpp"
#include "../math/Interpolation.hpp"
#include "Foreach.hpp"

namespace openpni::process::fbp {
// FBP处理参数结构体
struct FBPParam {
  std::size_t nRingDiff;
  double dBinMin;
  double dBinMax;
  int nSampNumInBin;
  int nSampNumInView;
  int nImgWidth;
  int nImgHeight;
  int nImgDepth;
  float voxelSizeXY;
  float voxelSizeZ;
  int deltalim;
  int klim;
  int wlim;
  double sampling_distance_in_s;
  double detectorLen;
};

// SSRB重组
struct SSRBProcessor {
  const float *michData; // 直接指向 MICH 数据的指针
  const example::PolygonalSystem &polygon;
  const basic::DetectorGeometry &detector;
  const FBPParam &params;

  float operator()(std::size_t idx) const;
};
struct FOREProcessor {
  using Complex = std::complex<double>;

  const float *michData;
  const example::PolygonalSystem &polygon;
  const basic::DetectorGeometry &detector;
  const FBPParam &params;

  // 静态变量存储FORE结果，确保只计算一次
  static std::unique_ptr<float[]> s_foreResult;
  static bool s_isComputed;
  static std::mutex s_computeMutex;
  static std::size_t s_resultSize;

  float operator()(std::size_t idx) const;

private:
  void computeFORE() const;
  std::vector<std::vector<float>> interpolateMatrix(const std::vector<std::vector<float>> &input_matrix,
                                                    int new_rows) const;
  std::vector<std::vector<double>> interpolateMatrixToTarget(const std::vector<std::vector<double>> &input_matrix,
                                                             int target_rows) const;
  std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>> &input) const;
  int next_power_of_2(int n) const;
  std::pair<std::vector<std::vector<std::complex<float>>>, std::vector<std::vector<float>>>
  fft_with_padding(const std::vector<std::vector<float>> &data) const;
  std::vector<std::vector<double>> ifft2(const std::vector<std::vector<Complex>> &input) const;
  std::vector<Complex> FFT(const std::vector<double> &input) const;
  std::vector<double> IFFT(const std::vector<Complex> &input) const;
  void FFTOrIFFT(std::vector<Complex> &data, bool inverse) const;
  std::vector<std::vector<Complex>> fft2D_efficient(const std::vector<std::vector<float>> &input) const;
  std::vector<std::vector<double>> ifft2D_efficient(const std::vector<std::vector<Complex>> &input) const;
  std::vector<std::vector<Complex>> ifft2D(const std::vector<std::vector<float>> &input, int rows, int cols) const;
  std::vector<std::vector<double>> ifft2D(const std::vector<std::vector<Complex>> &input) const;
  double linearInterpolate(const std::vector<double> &x, const std::vector<double> &y, double xi) const;
};

std::unique_ptr<float[]> FOREProcessor::s_foreResult = nullptr;
bool FOREProcessor::s_isComputed = false;
std::mutex FOREProcessor::s_computeMutex;
std::size_t FOREProcessor::s_resultSize = 0;

struct RealCoordGene {
  const example::PolygonalSystem &polygon;
  const basic::DetectorGeometry &detector;

  std::pair<float, float> operator()(std::size_t idx) const;
};

struct RegularMeshGenerator {
  const example::PolygonalSystem &polygon;
  const basic::DetectorGeometry &detector;
  const FBPParam &params;
  const RealCoordGene &realCoordGen;
  float thetaMin;
  float thetaMax;

  std::pair<float, float> operator()(std::size_t idx) const;
};

struct TriangleMapper {
  basic::Triangle<float> *triangleResults; // 预分配数组
  bool *foundFlags;                        // 标记是否找到
  const std::unique_ptr<float[]> &meshS;
  const std::unique_ptr<float[]> &meshTheta;
  const std::unique_ptr<basic::Triangle<float>[]> &triangles;
  std::size_t triangleCount;
  const FBPParam &params;

  float operator()(std::size_t idx) const;

  int binarySearch(const std::unique_ptr<basic::Triangle<float>[]> &triangles, std::size_t triangleCount,
                   float targetX) const;
};

struct BarycentricInterpolator {
  float *arcCorr;
  const std::unordered_map<basic::Point_2, basic::Triangle<float>> &triangleMap;
  const std::unordered_map<basic::Point_2, basic::Point_2> &values;
  const std::unique_ptr<float[]> &meshS;
  const std::unique_ptr<float[]> &meshTheta;
  const std::unique_ptr<float[]> &michRebin;
  const FBPParam &params;
  std::size_t nRebinSliceNum;
  const example::PolygonalSystem &polygon;
  const basic::DetectorGeometry &detector;

  float operator()(std::size_t idx) const;
};

struct BackProjectionCore {
  float *reconImg;
  const float *filterSino;
  const float *meshTheta;
  const float *meshImgX;
  const float *meshImgY;
  const FBPParam &params;
  std::size_t nRebinSliceNum;
  float axisSStart;
  float axisGap;

  void operator()(std::size_t globalIdx) const;
};

// FBP 函数 - 仅提供 SSRB 和 FORE 两种特定实例化
void FBP_SSRB(const FBPParam &params, const float *michData, const example::PolygonalSystem &polygon,
              const basic::DetectorGeometry &detector, float *outputImage,
              const basic::Image3DGeometry &outputGeometry);

void FBP_FORE(const FBPParam &params, const float *michData, const example::PolygonalSystem &polygon,
              const basic::DetectorGeometry &detector, float *outputImage,
              const basic::Image3DGeometry &outputGeometry);

} // namespace openpni::process::fbp

// #include "../Undef.h"