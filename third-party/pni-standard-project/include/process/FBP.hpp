#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <numbers>
#include <stdio.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../basic/Image.hpp"
#include "../basic/Triangulation.hpp"
#include "../detector/Detectors.hpp"
#include "../example/PolygonalSystem.hpp"
#include "../math/FouriorFilter.cuh"
#include "../math/Interpolation.hpp"
#include "Foreach.hpp"
// #include "../math/PointWise.hpp"
#include "../PnI-Config.hpp"

namespace openpni::process::fbp {
void saveComplexData(const std::string &filename,
                     const std::vector<std::vector<std::vector<std::complex<double>>>> &data);
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
template <typename T>
struct SSRBProcessor {
  using CalculatePrecision = T;

  const T *michData; // 直接指向 MICH 数据的指针
  const example::PolygonalSystem &polygon;
  const basic::DetectorGeometry &detector;
  const FBPParam &params;

  CalculatePrecision operator()(
      std::size_t idx) const {
    const auto nBinNum = example::polygon::getBinNum(polygon, detector);
    const auto nViewNum = example::polygon::getViewNum(polygon, detector);
    const auto nRingNum = example::polygon::getRingNum(polygon, detector);

    const auto sliceId = idx / (nViewNum * nBinNum);
    const auto remainIdx = idx % (nViewNum * nBinNum);
    const auto viewId = remainIdx / nBinNum;
    const auto binId = remainIdx % nBinNum;

    CalculatePrecision sum = 0.0f;
    int count = 0;
    const int hBound = sliceId < nRingNum ? sliceId : nRingNum - 1;
    const int lBound = sliceId >= nRingNum ? sliceId - nRingNum + 1 : 0;
    for (int ring0 = lBound; ring0 <= hBound; ring0++) {
      const int ring1 = sliceId - ring0;
      if (abs(ring0 - ring1) > static_cast<int>(params.nRingDiff))
        continue;
      std::size_t slice = ring0 * nRingNum + ring1;
      std::size_t lorIndex = slice * nBinNum * nViewNum + viewId * nBinNum + binId;

      sum += michData[lorIndex]; // 直接从指针访问数据
      count++;
    }
    return count > 0 ? sum / count : 0.0f;
  }
};
template <typename T>
struct FOREProcessor {
  using CalculatePrecision = T;
  using Complex = std::complex<double>;

  const T *michData;
  const example::PolygonalSystem &polygon;
  const basic::DetectorGeometry &detector;
  const FBPParam &params;

  // 静态变量存储FORE结果，确保只计算一次
  static std::unique_ptr<CalculatePrecision[]> s_foreResult;
  static bool s_isComputed;
  static std::mutex s_computeMutex;
  static std::size_t s_resultSize;

  CalculatePrecision operator()(
      std::size_t idx) const {
    if (!s_isComputed) {
      std::lock_guard<std::mutex> lock(s_computeMutex);
      if (!s_isComputed) {
        computeFORE();
        s_isComputed = true;
      }
    }
    return s_foreResult[idx];
  }

private:
  void computeFORE() const {
    const auto nBinNum = example::polygon::getBinNum(polygon, detector);
    const auto nViewNum = example::polygon::getViewNum(polygon, detector);
    const auto nRingNum = example::polygon::getRingNum(polygon, detector);
    const auto nRebinSliceNum = 2 * nRingNum - 1;
    s_resultSize = nRebinSliceNum * nViewNum * nBinNum;
    s_foreResult = std::make_unique<CalculatePrecision[]>(s_resultSize);

    // FORE参数初始化
    int viewPow = 2 * static_cast<int>(pow(2, std::ceil(std::log2(nViewNum))));
    int binPow = static_cast<int>(pow(2, std::ceil(std::log2(nBinNum))));
    double unitSpacing = params.detectorLen / nRingNum;
    int klim = params.klim;
    int wlim = params.wlim;
    double sampling_freq_w = 2 * M_PI / (params.sampling_distance_in_s * binPow);
    double R_field_of_view_mm = (nBinNum / 2 - 1) * params.sampling_distance_in_s;
    int deltalim = params.deltalim;
    double ringRadius = polygon.radius;

    std::cout << "FORE参数初始化完成" << std::endl;
    std::cout << "nBinNum: " << nBinNum << ", nViewNum: " << nViewNum << ", nRingNum: " << nRingNum << std::endl;
    std::cout << "viewPow: " << viewPow << ", binPow: " << binPow << std::endl;
    std::cout << "nRebinSliceNum: " << nRebinSliceNum << std::endl;
    std::cout << "sampling_freq_w: " << sampling_freq_w << std::endl;
    std::cout << "R_field_of_view_mm: " << R_field_of_view_mm << std::endl;
    std::cout << "unitSpacing: " << unitSpacing << std::endl;

    // Step 1: 创建p_z_delta数组
    std::vector<std::vector<std::vector<std::vector<double>>>> p_z_delta(
        nRingNum,
        std::vector<std::vector<std::vector<double>>>(
            2 * params.nRingDiff + 1, std::vector<std::vector<double>>(nViewNum, std::vector<double>(nBinNum, 0.0))));

// Step 2: 填充p_z_delta (并行化)
#pragma omp parallel for
    for (int ring1 = 0; ring1 < nRingNum; ++ring1) {
      for (int ring2 = 0; ring2 < nRingNum; ++ring2) {
        int z = static_cast<int>((ring1 + ring2) / 2);
        int delta = ring1 - ring2 + params.nRingDiff;

        if (abs(ring1 - ring2) <= static_cast<int>(params.nRingDiff)) {
          for (int view = 0; view < nViewNum; ++view) {
            for (int bin = 0; bin < nBinNum; ++bin) {
              std::size_t dataIndex = (ring1 + ring2 * nRingNum) * nViewNum * nBinNum + view * nBinNum + bin;
              p_z_delta[z][delta][view][bin] += michData[dataIndex];
            }
          }
        }
      }
    }

    // Step 3: FFT处理和FORE算法
    std::vector<std::vector<std::vector<Complex>>> FT_rebinned_data(
        nRebinSliceNum, std::vector<std::vector<Complex>>(binPow, std::vector<Complex>(viewPow, {0.0, 0.0})));

    std::vector<std::vector<std::vector<Complex>>> Weight_rebinned_data(
        nRebinSliceNum, std::vector<std::vector<Complex>>(binPow, std::vector<Complex>(viewPow, {0.0, 0.0})));

    // 处理每个ring difference
    for (int diff = 0; diff <= static_cast<int>(params.nRingDiff); ++diff) {
      // 创建组合数据
      std::vector<std::vector<std::vector<float>>> p_z_delta_Combined(
          nRingNum - diff, std::vector<std::vector<float>>(2 * nViewNum, std::vector<float>(nBinNum, 0.0)));

      for (int z = diff / 2; z < nRingNum - (diff + 1) / 2; ++z) {
        for (int view = 0; view < nViewNum; ++view) {
          for (int bin = 0; bin < nBinNum; ++bin) {
            p_z_delta_Combined[z - diff / 2][view][bin] = p_z_delta[z][params.nRingDiff - diff][view][bin];
            p_z_delta_Combined[z - diff / 2][nViewNum + view][bin] =
                p_z_delta[z][params.nRingDiff + diff][view][nBinNum - 1 - bin];
          }
        }
      }

      // FFT处理每个切片
      for (int z = 0; z < nRingNum - diff; ++z) {
        // 插值到viewPow大小
        auto interpolated = interpolateMatrix(p_z_delta_Combined[z], viewPow);

        // 进行2D FFT
        // 解构 fft_with_padding 的返回值
        auto [fft_result, padded_data] = fft_with_padding(transpose(interpolated));

        // FORE算法核心部分
        double t = diff * unitSpacing / ringRadius / 2.0;
        double z_in_mm = z * unitSpacing + diff * unitSpacing / 2.0;
        int z_index = static_cast<int>(round(z_in_mm / (unitSpacing / 2.0)));

        // 频域处理
        for (int j = wlim; j <= binPow / 2; ++j) {
          for (int i = klim; i <= viewPow / 2; ++i) {
            double w = j * sampling_freq_w;
            double k = i;
            if (k > w * R_field_of_view_mm)
              continue;

            double zShift = t * k / w;
            for (int direct : {-1, 1}) {
              int jj = (direct == -1) ? j : ((j > 0) ? binPow - j : j);
              double newZ = z_index + direct * zShift / (unitSpacing / 2);
              int smallZ = static_cast<int>(newZ);
              float m = newZ - smallZ;

              // 分布到相邻切片
              if (smallZ >= 0 && smallZ <= nRebinSliceNum - 1) {
                float OneMinusM = 1.0 - m;
                FT_rebinned_data[smallZ][jj][i] += fft_result[jj][i] * OneMinusM;
                Weight_rebinned_data[smallZ][jj][i] += OneMinusM;
              }
              if (smallZ >= -1 && smallZ < nRebinSliceNum - 1) {
                FT_rebinned_data[smallZ + 1][jj][i] += fft_result[jj][i] * m;
                Weight_rebinned_data[smallZ + 1][jj][i] += m;
              }
            }
          }
        }

        // 低频处理
        if (diff <= deltalim) {
          for (int j = 0; j < wlim; ++j) {
            for (int i = 0; i <= viewPow / 2; ++i) {
              for (int direct : {-1, 1}) {
                int jj = (direct == -1) ? j : ((j > 0) ? binPow - j : j);
                if (z_index >= 0 && z_index <= nRebinSliceNum - 1) {
                  FT_rebinned_data[z_index][jj][i] += fft_result[jj][i];
                  Weight_rebinned_data[z_index][jj][i] += 1.0;
                }
              }
            }
          }
          for (int j = wlim; j <= binPow / 2; ++j) {
            for (int i = 0; i <= klim; ++i) {
              for (int direct : {-1, 1}) {
                int jj = (direct == -1) ? j : ((j > 0) ? binPow - j : j);
                if (z_index >= 0 && z_index <= nRebinSliceNum - 1) {
                  FT_rebinned_data[z_index][jj][i] += fft_result[jj][i];
                  Weight_rebinned_data[z_index][jj][i] += 1.0;
                }
              }
            }
          }
        }
      }
    }

    // 在FORE函数中保存FT_rebinned_data和Weight_rebinned_data
    // openpni::process::fbp::saveComplexData("FT_rebinned_data", FT_rebinned_data);
    // openpni::process::fbp::saveComplexData("Weight_rebinned_data", Weight_rebinned_data);

    // Step 4: 权重处理和逆FFT
    std::vector<std::vector<std::vector<Complex>>> FT_Weight_data(
        nRebinSliceNum, std::vector<std::vector<Complex>>(binPow, std::vector<Complex>(viewPow / 2, {0.0, 0.0})));

#pragma omp parallel for
    for (int plane = 0; plane < nRebinSliceNum; ++plane) {
      for (int j = 0; j < binPow; ++j) {
        for (int i = 0; i < viewPow / 2; ++i) {
          Complex actual_weight =
              (std::abs(Weight_rebinned_data[plane][j][i]) < 1e-3) ? 0.0 : 1.0 / Weight_rebinned_data[plane][j][i];
          FT_Weight_data[plane][j][i] = actual_weight * FT_rebinned_data[plane][j][i];
        }
      }
    }

    // 扩展到完整频谱
    std::vector<std::vector<std::vector<Complex>>> FT_Weight_data_expend(
        nRebinSliceNum, std::vector<std::vector<Complex>>(binPow, std::vector<Complex>(viewPow, {0.0, 0.0})));

#pragma omp parallel for
    for (int plane = 0; plane < nRebinSliceNum; ++plane) {
      for (int i = 0; i < binPow; ++i) {
        for (int j = 0; j < viewPow / 2; ++j) {
          FT_Weight_data_expend[plane][i][j] = FT_Weight_data[plane][i][j];
        }
        for (int j = viewPow / 2; j < viewPow; ++j) {
          if (i != 0 && j != viewPow / 2) {
            FT_Weight_data_expend[plane][i][j] = std::conj(FT_Weight_data[plane][binPow - i][viewPow - j]);
          } else if (i == 0 && j != viewPow / 2) {
            FT_Weight_data_expend[plane][i][j] = std::conj(FT_Weight_data[plane][0][viewPow - j]);
          }
        }
      }
    }

    // 逆FFT并最终插值
    for (int plane = 0; plane < nRebinSliceNum; ++plane) {
      auto real_data = ifft2D(FT_Weight_data_expend[plane]);

      // 裁剪到原始大小
      std::vector<std::vector<double>> cropped_data(nBinNum, std::vector<double>(viewPow, 0.0));
      for (int j = 0; j < nBinNum; ++j) {
        for (int k = 0; k < viewPow; ++k) {
          cropped_data[j][k] = real_data[j][k];
        }
      }

      // 转置
      std::vector<std::vector<double>> transposed_data(viewPow / 2, std::vector<double>(nBinNum, 0.0));
      for (int i = 0; i < viewPow / 2; ++i) {
        for (int j = 0; j < nBinNum; ++j) {
          transposed_data[i][j] = cropped_data[j][i];
        }
      }

      // 插值到目标视图数
      auto final_result = interpolateMatrixToTarget(transposed_data, nViewNum);

      // 存储最终结果
      for (int view = 0; view < nViewNum; ++view) {
        for (int bin = 0; bin < nBinNum; ++bin) {
          std::size_t resultIndex = plane * nViewNum * nBinNum + view * nBinNum + bin;
          s_foreResult[resultIndex] = static_cast<CalculatePrecision>(final_result[view][bin]);
        }
      }
    }
  }

  // 辅助函数实现
  std::vector<std::vector<float>> interpolateMatrix(
      const std::vector<std::vector<float>> &input_matrix, int new_rows) const {
    int original_rows = input_matrix.size();
    if (original_rows == 0) {
      throw std::invalid_argument("Input matrix cannot be empty.");
    }
    int num_cols = input_matrix[0].size();

    std::vector<std::vector<float>> interpolated_matrix(new_rows, std::vector<float>(num_cols, 0.0));

    for (int col = 0; col < num_cols; ++col) {
      for (int i = 0; i < new_rows; ++i) {
        double x = i * (static_cast<double>(original_rows - 1) / (new_rows - 1));
        int x0 = static_cast<int>(std::floor(x));
        int x1 = std::min(x0 + 1, original_rows - 1);

        double y0 = input_matrix[x0][col];
        double y1 = input_matrix[x1][col];

        if (x1 == x0) {
          interpolated_matrix[i][col] = y0;
        } else {
          interpolated_matrix[i][col] = y0 + (x - x0) * (y1 - y0) / (x1 - x0);
        }
      }
    }
    return interpolated_matrix;
  }

  std::vector<std::vector<double>> interpolateMatrixToTarget(
      const std::vector<std::vector<double>> &input_matrix, int target_rows) const {
    int original_rows = input_matrix.size();
    int num_cols = input_matrix[0].size();

    std::vector<std::vector<double>> result(target_rows, std::vector<double>(num_cols, 0.0));

    std::vector<double> x_original(original_rows);
    std::vector<double> x_new(target_rows);
    for (int i = 0; i < original_rows; ++i) {
      x_original[i] = static_cast<double>(i) / (original_rows - 1);
    }
    for (int i = 0; i < target_rows; ++i) {
      x_new[i] = static_cast<double>(i) / (target_rows - 1);
    }

    for (int col = 0; col < num_cols; ++col) {
      std::vector<double> column(original_rows);
      for (int i = 0; i < original_rows; ++i) {
        column[i] = input_matrix[i][col];
      }

      for (int i = 0; i < target_rows; ++i) {
        result[i][col] = linearInterpolate(x_original, column, x_new[i]);
      }
    }
    return result;
  }

  std::vector<std::vector<float>> transpose(
      const std::vector<std::vector<float>> &input) const {
    if (input.empty() || input[0].empty()) {
      return {};
    }

    int rows = input.size();
    int cols = input[0].size();
    std::vector<std::vector<float>> transposed(cols, std::vector<float>(rows, 0.0));

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        transposed[j][i] = input[i][j];
      }
    }
    return transposed;
  }

  // 从您提供的源代码中整合的辅助函数
  int next_power_of_2(
      int n) const {
    if (n <= 0)
      return 1;
    return pow(2, ceil(log2(n)));
  }

  // 完整的FFT实现
  std::pair<std::vector<std::vector<std::complex<float>>>, std::vector<std::vector<float>>> fft_with_padding(
      const std::vector<std::vector<float>> &data) const {
    // 获取输入矩阵的形状
    int rows = data.size();
    int cols = data[0].size();

    // 计算填充后的尺寸（下一次 2 的幂次方）
    int new_rows = next_power_of_2(rows);
    int new_cols = next_power_of_2(cols);

    // 进行零填充
    std::vector<std::vector<float>> padded_data(new_rows, std::vector<float>(new_cols, 0.0f));
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        padded_data[i][j] = data[i][j];
      }
    }

    // 创建 FFT 输入和输出数组
    fftw_complex *fft_in = fftw_alloc_complex(new_rows * new_cols);
    fftw_complex *fft_out = fftw_alloc_complex(new_rows * new_cols);

    // 填充 FFT 输入数组
    for (int i = 0; i < new_rows; ++i) {
      for (int j = 0; j < new_cols; ++j) {
        int index = i * new_cols + j;
        fft_in[index][0] = padded_data[i][j]; // 实部
        fft_in[index][1] = 0.0;               // 虚部
      }
    }

    // 创建 FFTW 计划并执行 FFT
    fftw_plan plan = fftw_plan_dft_2d(new_rows, new_cols, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // 将 FFT 结果转换为二维复数矩阵
    std::vector<std::vector<std::complex<float>>> fft_result(new_rows, std::vector<std::complex<float>>(new_cols));
    for (int i = 0; i < new_rows; ++i) {
      for (int j = 0; j < new_cols; ++j) {
        int index = i * new_cols + j;
        fft_result[i][j] = std::complex<float>(fft_out[index][0], fft_out[index][1]);
      }
    }

    // 清理 FFTW 资源
    fftw_destroy_plan(plan);
    fftw_free(fft_in);
    fftw_free(fft_out);

    return {fft_result, padded_data};
  }

  std::vector<std::vector<double>> ifft2(
      const std::vector<std::vector<Complex>> &input) const {
    int rows = input.size();
    int cols = input[0].size();

    // 创建 FFTW 输入和输出数组
    fftw_complex *fft_in = fftw_alloc_complex(rows * cols);
    fftw_complex *fft_out = fftw_alloc_complex(rows * cols);

    // 填充 FFT 输入数组
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        int index = i * cols + j;
        fft_in[index][0] = input[i][j].real(); // 实部
        fft_in[index][1] = input[i][j].imag(); // 虚部
      }
    }

    // 创建 FFTW 计划并执行 IFFT
    fftw_plan plan = fftw_plan_dft_2d(rows, cols, fft_in, fft_out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // 将 IFFT 结果转换为二维实数矩阵，并归一化
    std::vector<std::vector<double>> result(rows, std::vector<double>(cols, 0.0));
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        int index = i * cols + j;
        result[i][j] = fft_out[index][0] / (rows * cols); // 归一化
      }
    }

    // 清理 FFTW 资源
    fftw_destroy_plan(plan);
    fftw_free(fft_in);
    fftw_free(fft_out);

    return result;
  }

  // 1D FFT实现
  std::vector<Complex> FFT(
      const std::vector<double> &input) const {
    int nlen = input.size();
    if ((nlen > 0) && (nlen & (nlen - 1)) != 0) {
      printf("Error! At FOREProcessor::FFT. Invalid input\n");
      return std::vector<Complex>{};
    }

    std::vector<Complex> data(nlen);
    for (int i = 0; i < nlen; ++i) {
      data[i] = Complex(input[i], 0.0);
    }
    FFTOrIFFT(data, false);
    return data;
  }

  std::vector<double> IFFT(
      const std::vector<Complex> &input) const {
    int nlen = input.size();
    if ((nlen > 0) && (nlen & (nlen - 1)) != 0) {
      printf("Error! At FOREProcessor::IFFT. Invalid input\n");
      return std::vector<double>{};
    }

    std::vector<Complex> data = input;
    FFTOrIFFT(data, true);
    std::vector<double> output(nlen);
    for (int i = 0; i < nlen; ++i) {
      output[i] = data[i].real();
    }
    return output;
  }

  void FFTOrIFFT(
      std::vector<Complex> &data, bool inverse) const {
    int nLen = data.size();
    if (nLen <= 1)
      return;

    // 递归进行FFT或IFFT变换
    std::vector<Complex> even(nLen / 2), odd(nLen / 2);
    for (int i = 0; i < nLen / 2; ++i) {
      even[i] = data[2 * i];
      odd[i] = data[2 * i + 1];
    }
    FFTOrIFFT(even, inverse);
    FFTOrIFFT(odd, inverse);

    for (int k = 0; k < nLen / 2; ++k) {
      Complex t = std::exp(Complex(0, (inverse ? 2 : -2) * M_PI * k / nLen)) * odd[k];
      data[k] = even[k] + t;
      data[k + nLen / 2] = even[k] - t;
    }

    // 对于IFFT，还需要除以2
    if (inverse) {
      for (int i = 0; i < nLen; ++i) {
        data[i] /= 2.0;
      }
    }
  }

  // 使用FFTW的更高效2D FFT实现
  std::vector<std::vector<Complex>> fft2D_efficient(
      const std::vector<std::vector<float>> &input) const {
    int rows = input.size();
    int cols = input[0].size();

    // 分配FFTW内存
    fftw_complex *in = fftw_alloc_complex(rows * cols);
    fftw_complex *out = fftw_alloc_complex(rows * cols);

    // 填充输入数据
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        int idx = i * cols + j;
        in[idx][0] = input[i][j]; // 实部
        in[idx][1] = 0.0;         // 虚部
      }
    }

    // 创建2D FFT计划
    fftw_plan plan = fftw_plan_dft_2d(rows, cols, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // 执行FFT
    fftw_execute(plan);

    // 转换结果
    std::vector<std::vector<Complex>> result(rows, std::vector<Complex>(cols));
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        int idx = i * cols + j;
        result[i][j] = Complex(out[idx][0], out[idx][1]);
      }
    }

    // 清理资源
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return result;
  }

  // 使用FFTW的更高效2D IFFT实现
  std::vector<std::vector<double>> ifft2D_efficient(
      const std::vector<std::vector<Complex>> &input) const {
    int rows = input.size();
    int cols = input[0].size();

    // 分配FFTW内存
    fftw_complex *in = fftw_alloc_complex(rows * cols);
    fftw_complex *out = fftw_alloc_complex(rows * cols);

    // 填充输入数据
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        int idx = i * cols + j;
        in[idx][0] = input[i][j].real();
        in[idx][1] = input[i][j].imag();
      }
    }

    // 创建2D IFFT计划
    fftw_plan plan = fftw_plan_dft_2d(rows, cols, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

    // 执行IFFT
    fftw_execute(plan);

    // 转换结果并归一化
    std::vector<std::vector<double>> result(rows, std::vector<double>(cols));
    double normalize_factor = 1.0 / (rows * cols);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        int idx = i * cols + j;
        result[i][j] = out[idx][0] * normalize_factor;
      }
    }

    // 清理资源
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return result;
  }

  // 修改fft2D函数使用高效版本
  std::vector<std::vector<Complex>> ifft2D(
      const std::vector<std::vector<float>> &input, int rows, int cols) const {
    // 如果需要填充，先填充再FFT
    if (input.size() != rows || input[0].size() != cols) {
      auto [fft_result, padded_data] = fft_with_padding(input);
      return std::vector<std::vector<Complex>>(fft_result.begin(), fft_result.end());
    } else {
      return fft2D_efficient(input);
    }
  }

  // 修改ifft2D函数使用高效版本
  std::vector<std::vector<double>> ifft2D(
      const std::vector<std::vector<Complex>> &input) const {
    return ifft2D_efficient(input);
  }

  double linearInterpolate(
      const std::vector<double> &x, const std::vector<double> &y, double xi) const {
    if (x.size() != y.size() || x.empty()) {
      throw std::invalid_argument("x and y must have the same non-zero size.");
    }

    auto it = std::lower_bound(x.begin(), x.end(), xi);
    if (it == x.begin())
      return y.front();
    if (it == x.end())
      return y.back();

    size_t idx = std::distance(x.begin(), it) - 1;
    double x1 = x[idx], x2 = x[idx + 1];
    double y1 = y[idx], y2 = y[idx + 1];

    if (std::abs(x2 - x1) < 1e-6) {
      return (y1 + y2) / 2.0;
    }

    return y1 + (y2 - y1) * (xi - x1) / (x2 - x1);
  }

  // Helper function to save 3D complex data to a file
};

// 静态成员定义
template <typename T>
std::unique_ptr<typename FOREProcessor<T>::CalculatePrecision[]> FOREProcessor<T>::s_foreResult = nullptr;

template <typename T>
bool FOREProcessor<T>::s_isComputed = false;

template <typename T>
std::mutex FOREProcessor<T>::s_computeMutex;

template <typename T>
std::size_t FOREProcessor<T>::s_resultSize = 0;

template <FloatingPoint_c ValueType>
struct RealCoordGene {
  const example::PolygonalSystem &polygon;
  const basic::DetectorGeometry &detector;
  std::pair<ValueType, ValueType> operator()(
      std::size_t idx) const {
    const auto nBinNum = example::polygon::getBinNum(polygon, detector);
    const auto nViewNum = example::polygon::getViewNum(polygon, detector);
    const int view = static_cast<int>(idx) / static_cast<int>(nBinNum);
    const int bin = static_cast<int>(idx) % static_cast<int>(nBinNum);

    int crystalOneRing = example::polygon::getCrystalNumOneRing(polygon, detector);
    int cry2 = bin / 2 + 1;
    int cry1 = crystalOneRing + (1 - bin % 2) - cry2;
    cry2 = (cry2 + view) % crystalOneRing;
    cry1 = (cry1 + view) % crystalOneRing;

    // 将矩形坐标系的晶体ID转换为uniform坐标系
    uint32_t uniformCry1 = example::polygon::getUniformIDFromRectangleID(polygon, detector, cry1);
    uint32_t uniformCry2 = example::polygon::getUniformIDFromRectangleID(polygon, detector, cry2);

    // 计算探测器参数
    const auto totalDetectorNum = polygon.totalDetectorNum();
    const auto crystalNumInDetector = detector.getTotalCrystalNum();

    // 现在使用uniform坐标系的ID计算探测器索引
    unsigned detectorIndex1 = uniformCry1 / crystalNumInDetector;
    unsigned crystalIndex1 = uniformCry1 % crystalNumInDetector;
    unsigned detectorIndex2 = uniformCry2 / crystalNumInDetector;
    unsigned crystalIndex2 = uniformCry2 % crystalNumInDetector;

    // 获取探测器坐标
    auto detectorCoord1 =
        example::coordinateFromPolygon(polygon, detectorIndex1 / (polygon.detectorPerEdge * polygon.edges),
                                       detectorIndex1 % (polygon.detectorPerEdge * polygon.edges));

    auto detectorCoord2 =
        example::coordinateFromPolygon(polygon, detectorIndex2 / (polygon.detectorPerEdge * polygon.edges),
                                       detectorIndex2 % (polygon.detectorPerEdge * polygon.edges));

    // 计算晶体位置
    auto crystalCoord1 = basic::calculateCrystalGeometry(detectorCoord1, detector, crystalIndex1);
    auto crystalCoord2 = basic::calculateCrystalGeometry(detectorCoord2, detector, crystalIndex2);

    auto point1 = crystalCoord1.position;
    auto point2 = crystalCoord2.position;

    // 计算thetaReal
    ValueType thetaReal = std::atan2(point2.y - point1.y, point2.x - point1.x) * ValueType(180) / ValueType(M_PI);

    if (thetaReal < 0)
      thetaReal += ValueType(360);
    // 计算sReal - 直接内联两点连线到原点的距离计算
    int nsign = (bin > static_cast<int>(nBinNum) / 2) ? 1 : -1;

    // 计算直线的A,B,C参数：Ax + By + C = 0
    double A = point2.y - point1.y;
    double B = point1.x - point2.x;
    double C = point2.x * point1.y - point1.x * point2.y;

    ValueType sReal;
    if (A == 0 && B == 0) {
      sReal = ValueType(0);
    } else {
      sReal = nsign * ValueType(std::abs(C) / std::sqrt(A * A + B * B));
    }

    if (bin == static_cast<int>(nBinNum) / 2)
      sReal = ValueType(0);
    return std::make_pair(sReal, thetaReal);
  }
};

template <FloatingPoint_c ValueType>
struct RegularMeshGenerator {
  const example::PolygonalSystem &polygon;
  const basic::DetectorGeometry &detector;
  const FBPParam &params;
  const RealCoordGene<ValueType> &realCoordGen;
  ValueType thetaMin;
  ValueType thetaMax;

  std::pair<ValueType, ValueType> operator()(
      std::size_t idx) const {
    const auto nBinNum = example::polygon::getBinNum(polygon, detector);
    const auto nViewNum = example::polygon::getViewNum(polygon, detector);
    const int viewIdx = static_cast<int>(idx) / params.nSampNumInBin;
    const int binIdx = static_cast<int>(idx) % params.nSampNumInBin;

    ValueType gridS = ValueType(params.dBinMin) +
                      binIdx * ValueType(params.dBinMax - params.dBinMin) / ValueType(params.nSampNumInBin);
    ValueType gridTheta = thetaMin + viewIdx * (thetaMax - thetaMin) / ValueType(params.nSampNumInView);
    ValueType deltaS = ValueType(params.dBinMax - params.dBinMin) / ValueType(params.nSampNumInBin);
    gridS += deltaS / ValueType(2);
    ValueType deltaTheta = (thetaMax - thetaMin) / ValueType(params.nSampNumInView);
    gridTheta += deltaTheta / ValueType(2);
    return std::make_pair(gridS, gridTheta);
  }
};

template <FloatingPoint_c ValueType>
struct TriangleMapper {
  basic::Triangle<ValueType> *triangleResults; // 预分配数组
  bool *foundFlags;                            // 标记是否找到
  const std::unique_ptr<ValueType[]> &meshS;
  const std::unique_ptr<ValueType[]> &meshTheta;
  const std::unique_ptr<basic::Triangle<ValueType>[]> &triangles;
  std::size_t triangleCount;
  const FBPParam &params;

  ValueType operator()(
      std::size_t idx) const {
    const int bin = static_cast<int>(idx) / static_cast<int>(params.nSampNumInView);
    const int view = static_cast<int>(idx) % static_cast<int>(params.nSampNumInView);

    std::size_t meshIdx = view * params.nSampNumInBin + bin;
    ValueType x = meshS[meshIdx];
    ValueType y = meshTheta[meshIdx];
    basic::Point_2 queryPoint(x, y);

    int searchIdx = binarySearch(triangles, triangleCount, x);

    foundFlags[idx] = false; // 无锁写入
    for (int i = searchIdx; i >= 0; i--) {
      if (triangles[i].IsInside(queryPoint)) {
        triangleResults[idx] = triangles[i]; // 无锁写入
        foundFlags[idx] = true;
        break;
      }
    }
    return static_cast<ValueType>(0.0f);
  }

  int binarySearch(
      const std::unique_ptr<basic::Triangle<ValueType>[]> &triangles, std::size_t triangleCount,
      ValueType targetX) const {
    int left = 0, right = static_cast<int>(triangleCount) - 1;

    while (left <= right) {
      if (left == right)
        break;

      int mid = (left + right + 1) >> 1;
      if (targetX <= triangles[mid].minX()) {
        right = mid - 1;
      } else {
        left = mid;
      }
    }
    return (left + 1 < static_cast<int>(triangleCount)) ? left + 1 : left;
  }
};

template <FloatingPoint_c ValueType>
struct BarycentricInterpolator {
  float *arcCorr;
  const std::unordered_map<basic::Point_2, basic::Triangle<ValueType>> &triangleMap;
  const std::unordered_map<basic::Point_2, basic::Point_2> &values;
  const std::unique_ptr<ValueType[]> &meshS;
  const std::unique_ptr<ValueType[]> &meshTheta;
  const std::unique_ptr<float[]> &michRebin;
  const FBPParam &params;
  std::size_t nRebinSliceNum;
  const example::PolygonalSystem &polygon;
  const basic::DetectorGeometry &detector;

  float operator()(
      std::size_t idx) const {
    const int i = static_cast<int>(idx) / static_cast<int>(params.nSampNumInView * params.nSampNumInBin);
    const std::size_t remainingIdx = idx % static_cast<std::size_t>(params.nSampNumInBin * params.nSampNumInView);
    const int bin = static_cast<int>(remainingIdx) / static_cast<int>(params.nSampNumInView);
    const int view = static_cast<int>(remainingIdx) % static_cast<int>(params.nSampNumInView);

    std::size_t meshIdx = view * params.nSampNumInBin + bin;
    ValueType x = meshS[meshIdx];
    ValueType y = meshTheta[meshIdx];
    basic::Point_2 queryPoint(x, y);

    const auto iter = triangleMap.find(queryPoint);
    if (iter == triangleMap.end()) {
      // std::cout << "Warning: No triangle found for point (" << x << ", " << y
      //           << "). Using default value." << std::endl;
      // std::cout << "meshIdx: " << meshIdx << std::endl;
      std::size_t arcCorrIdx = i * (params.nSampNumInView * params.nSampNumInBin) + view * params.nSampNumInBin + bin;
      arcCorr[arcCorrIdx] = 0.0f;
      return static_cast<ValueType>(0.0f);
    }
    basic::Triangle<ValueType> triangle = iter->second;
    const auto nBinNum = example::polygon::getBinNum(this->polygon, this->detector);
    const auto nViewNum = example::polygon::getViewNum(this->polygon, this->detector);

    // 使用at()方法进行安全访问，或者使用find()方法
    auto findP1 = values.find(triangle.p1);
    auto findP2 = values.find(triangle.p2);
    auto findP3 = values.find(triangle.p3);

    if (findP1 != values.end() && findP2 != values.end() && findP3 != values.end()) {
      triangle.v1 = michRebin[i * (nViewNum * nBinNum) + int(findP1->second.x()) * nBinNum + int(findP1->second.y())];
      triangle.v2 = michRebin[i * (nViewNum * nBinNum) + int(findP2->second.x()) * nBinNum + int(findP2->second.y())];
      triangle.v3 = michRebin[i * (nViewNum * nBinNum) + int(findP3->second.x()) * nBinNum + int(findP3->second.y())];
    } else {
      triangle.v1 = triangle.v2 = triangle.v3 = 0.0f;
    }

    std::size_t arcCorrIdx = i * (params.nSampNumInView * params.nSampNumInBin) + view * params.nSampNumInBin + bin;
    arcCorr[arcCorrIdx] = triangle.BarycentricInterpolationValue(queryPoint);
    return static_cast<ValueType>(0.0f);
  }
};

template <FloatingPoint_c ValueType>
struct BackProjectionCore {
  ValueType *reconImg;
  const ValueType *filterSino;
  const ValueType *meshTheta;
  const ValueType *meshImgX;
  const ValueType *meshImgY;
  const FBPParam &params;
  std::size_t nRebinSliceNum;
  ValueType axisSStart;
  ValueType axisGap;

  void operator()(
      std::size_t globalIdx) const {
    const std::size_t pointsPerSlice = params.nSampNumInView * params.nImgWidth * params.nImgHeight;
    const std::size_t i = globalIdx / pointsPerSlice;
    const std::size_t remainingIdx = globalIdx % pointsPerSlice;

    const std::size_t pointsPerView = params.nImgWidth * params.nImgHeight;
    const int thetaID = static_cast<int>(remainingIdx) / static_cast<int>(pointsPerView);
    const std::size_t pixelIdx = remainingIdx % pointsPerView;

    const int mx = static_cast<int>(pixelIdx) / params.nImgHeight;
    const int my = static_cast<int>(pixelIdx) % params.nImgHeight;
    std::size_t thetaIdx = thetaID * params.nSampNumInBin;
    ValueType theta = meshTheta[thetaIdx] * ValueType(M_PI) / ValueType(180);
    ValueType sinTheta = std::sin(theta);
    ValueType cosTheta = std::cos(theta);

    std::size_t imgIdx = my * params.nImgWidth + mx;
    ValueType binPosition = meshImgY[imgIdx] * cosTheta + meshImgX[imgIdx] * sinTheta;

    int valueIndexLeft = static_cast<int>((binPosition - axisSStart) / axisGap);
    int valueIndexRight = valueIndexLeft + 1;
    std::size_t leftIdx =
        i * (params.nSampNumInView * params.nSampNumInBin) + thetaID * params.nSampNumInBin + valueIndexLeft;
    std::size_t rightIdx =
        i * (params.nSampNumInView * params.nSampNumInBin) + thetaID * params.nSampNumInBin + valueIndexRight;

    ValueType valueLeft = filterSino[leftIdx];
    ValueType valueRight = filterSino[rightIdx];
    ValueType interpolatedValue =
        valueLeft + (binPosition - axisSStart - valueIndexLeft * axisGap) / axisGap * (valueRight - valueLeft);

    std::size_t reconIdx = i * (params.nImgWidth * params.nImgHeight) + my * params.nImgWidth + mx;
    reconImg[reconIdx] += interpolatedValue;
  }
};

template <typename RebinMethod>
void FBP(
    const FBPParam &params, const typename RebinMethod::CalculatePrecision *michData,
    const example::PolygonalSystem &polygon, const basic::DetectorGeometry &detector, float *outputImage,
    const basic::Image3DGeometry &outputGeometry) {
  using ValueType = typename RebinMethod::CalculatePrecision;
  unsigned int thread_num = std::thread::hardware_concurrency();

  // 获取几何参数
  const auto nBinNum = example::polygon::getBinNum(polygon, detector);
  const auto nViewNum = example::polygon::getViewNum(polygon, detector);
  const auto nRingNum = example::polygon::getRingNum(polygon, detector);

  // Step 1: 傅里叶重组 (Fourier Rebinning)
  const auto nRebinSliceNum = 2 * nRingNum - 1;
  const auto rebinDataSize = nRebinSliceNum * nViewNum * nBinNum;
  auto michRebin = std::make_unique<ValueType[]>(rebinDataSize);
  RebinMethod rebinProcessor{michData, polygon, detector, params};
  process::for_each(
      rebinDataSize, [&](std::size_t i) { michRebin[i] = rebinProcessor(i); }, basic::CpuMultiThread(thread_num));

  {
    std::ofstream michRebinFile("michRebin.bin", std::ios::binary);
    michRebinFile.write(reinterpret_cast<const char *>(michRebin.get()), rebinDataSize * sizeof(ValueType));
    michRebinFile.close();
  }

  // Step 2: 弧度校正 (Arc Correction)
  const auto irregularPointsSize = nBinNum * nViewNum;
  // Step 1.5: 计算theta的实际范围
  std::cout << "计算theta范围..." << std::endl;
  ValueType thetaMin = std::numeric_limits<ValueType>::max();
  ValueType thetaMax = std::numeric_limits<ValueType>::lowest();

  const auto arcCorrSize = nRebinSliceNum * params.nSampNumInView * params.nSampNumInBin;
  auto arcCorr = std::make_unique<ValueType[]>(arcCorrSize);
  const auto meshSize = params.nSampNumInView * params.nSampNumInBin;
  auto meshS = std::make_unique<ValueType[]>(meshSize);
  auto meshTheta = std::make_unique<ValueType[]>(meshSize);
  RealCoordGene<ValueType> realCoordGen{polygon, detector};
  // 遍历所有点计算theta范围
  for (std::size_t i = 0; i < irregularPointsSize; ++i) {
    auto [sReal, thetaReal] = realCoordGen(i);
    thetaMin = std::min(thetaMin, thetaReal);
    thetaMax = std::max(thetaMax, thetaReal);
  }

  RegularMeshGenerator<ValueType> meshGen{polygon, detector, params, realCoordGen, thetaMin, thetaMax};

  process::for_each(meshSize, [&](std::size_t i) {
    auto [meshSVal, meshThetaVal] = meshGen(i);
    meshS[i] = meshSVal;
  });
  process::for_each(meshSize, [&](std::size_t i) {
    auto [meshSVal, meshThetaVal] = meshGen(i);
    meshTheta[i] = meshThetaVal;
  });

  auto points = std::make_unique<basic::Point_2[]>(irregularPointsSize);
  process::for_each(
      irregularPointsSize,
      [&](std::size_t i) {
        auto [sReal, thetaReal] = realCoordGen(i);
        points[i] = basic::Point_2(sReal, thetaReal);
      },
      basic::CpuMultiThread(thread_num));

  std::unordered_map<basic::Point_2, basic::Point_2> values;
  auto dummyOutput1 = std::make_unique<float[]>(irregularPointsSize);
  process::for_each(
      irregularPointsSize,
      [&](std::size_t i) {
        const int view = static_cast<int>(i) / static_cast<int>(nBinNum);
        const int bin = static_cast<int>(i) % static_cast<int>(nBinNum);
        values[points[i]] = basic::Point_2(view, bin);
        dummyOutput1[i] = ValueType(0);
      },
      basic::CpuMultiThread(1));
  std::size_t triangleCount = 0;
  auto triangles = basic::GetDelaunayTriangles<ValueType>(points, irregularPointsSize, triangleCount);
  std::sort(
      triangles.get(), triangles.get() + triangleCount,
      [](const basic::Triangle<ValueType> &a, const basic::Triangle<ValueType> &b) { return a.minX() < b.minX(); });
  const auto totalMeshPoints = params.nSampNumInView * params.nSampNumInBin;
  auto triangleResults = std::make_unique<basic::Triangle<ValueType>[]>(totalMeshPoints);
  auto foundFlags = std::make_unique<bool[]>(totalMeshPoints);
  TriangleMapper<ValueType> triangleMapper{
      triangleResults.get(), foundFlags.get(), meshS, meshTheta, triangles, triangleCount, params};
  auto dummyOutput2 = std::make_unique<float[]>(totalMeshPoints);
  process::for_each(
      totalMeshPoints, [&](std::size_t i) { dummyOutput2[i] = triangleMapper(i); }, basic::CpuMultiThread(thread_num));
  // 串行阶段：构建哈希表
  std::unordered_map<basic::Point_2, basic::Triangle<ValueType>> triangleMap;
  for (std::size_t idx = 0; idx < totalMeshPoints; ++idx) {
    if (foundFlags[idx]) {
      const int bin = static_cast<int>(idx) / static_cast<int>(params.nSampNumInView);
      const int view = static_cast<int>(idx) % static_cast<int>(params.nSampNumInView);
      std::size_t meshIdx = view * params.nSampNumInBin + bin;
      ValueType x = meshS[meshIdx];
      ValueType y = meshTheta[meshIdx];
      basic::Point_2 queryPoint(x, y);
      triangleMap[queryPoint] = triangleResults[idx];
    }
  }
  BarycentricInterpolator<ValueType> interpolator{arcCorr.get(), triangleMap, values,         meshS,   meshTheta,
                                                  michRebin,     params,      nRebinSliceNum, polygon, detector};
  const auto totalInterpolationPoints = nRebinSliceNum * params.nSampNumInView * params.nSampNumInBin;
  auto dummyOutput3 = std::make_unique<float[]>(totalInterpolationPoints);

  process::for_each(
      totalInterpolationPoints, [&](std::size_t i) { dummyOutput3[i] = interpolator(i); },
      basic::CpuMultiThread(thread_num));

  // Step 3: 滤波（Filter）- 直接复用FDK的CUDA滤波
  const auto filterSinoSize = nRebinSliceNum * params.nSampNumInView * params.nSampNumInBin;
  auto filterSino = std::make_unique<ValueType[]>(filterSinoSize);

  auto d_filter = openpni::make_cuda_sync_ptr<typename openpni::process::CuFFTPrecisionAdapter<ValueType>::cufft_type>(
      params.nSampNumInBin);
  process::initializeFouriorFilter<ValueType>(d_filter.get(), params.nSampNumInBin, params.nSampNumInView);
  process::FouriorCutoffFilter<ValueType> ff{d_filter.get(), static_cast<unsigned>(params.nSampNumInBin),
                                             static_cast<unsigned>(params.nSampNumInView), 250};
  size_t totalSize = params.nSampNumInBin * params.nSampNumInView * nRebinSliceNum;
  auto d_arcCorr = openpni::make_cuda_sync_ptr<ValueType>(totalSize);
  auto d_filterSino = openpni::make_cuda_sync_ptr<ValueType>(totalSize);
  cudaMemcpy(d_arcCorr.get(), arcCorr.get(), totalSize * sizeof(ValueType), cudaMemcpyHostToDevice);
  for (std::size_t i = 0; i < nRebinSliceNum; i++) {
    for (int view = 0; view < params.nSampNumInView; view++) {
      process::fouriorFilter(
          &d_arcCorr.get()[i * params.nSampNumInView * params.nSampNumInBin + view * params.nSampNumInBin],
          &d_filterSino.get()[i * params.nSampNumInView * params.nSampNumInBin + view * params.nSampNumInBin],
          basic::Vec2<unsigned>(params.nSampNumInBin, 1), ff);
    }
  }
  cudaMemcpy(filterSino.get(), d_filterSino.get(), totalSize * sizeof(ValueType), cudaMemcpyDeviceToHost);

  // Step 4: 反投影（BackProject）
  const auto reconImgSize = nRebinSliceNum * params.nImgWidth * params.nImgHeight;
  auto reconImg = std::make_unique<ValueType[]>(reconImgSize);
  std::fill(reconImg.get(), reconImg.get() + reconImgSize, ValueType(0));
  ValueType fovSizeX = params.nImgWidth * params.voxelSizeXY;
  ValueType fovSizeY = params.nImgHeight * params.voxelSizeXY;
  auto imgX = std::make_unique<ValueType[]>(params.nImgWidth);
  auto imgY = std::make_unique<ValueType[]>(params.nImgHeight);
  auto meshImgX = std::make_unique<ValueType[]>(params.nImgWidth * params.nImgHeight);
  auto meshImgY = std::make_unique<ValueType[]>(params.nImgHeight * params.nImgWidth);
  process::for_each(
      params.nImgWidth,
      [&](std::size_t i) {
        imgX[params.nImgWidth - 1 - i] = (i + ValueType(0.5)) * params.voxelSizeXY - fovSizeX / ValueType(2);
      },
      basic::CpuMultiThread(thread_num));
  process::for_each(
      params.nImgHeight,
      [&](std::size_t i) { imgY[i] = (i + ValueType(0.5)) * params.voxelSizeXY - fovSizeY / ValueType(2); },
      basic::CpuMultiThread(thread_num));
  process::for_each(params.nImgWidth * params.nImgHeight, [&](std::size_t idx) {
    int i = static_cast<int>(idx) / params.nImgHeight;
    int j = static_cast<int>(idx) % params.nImgHeight;
    meshImgX[j * params.nImgWidth + i] = imgX[i];
    meshImgY[j * params.nImgWidth + i] = imgY[j];
  });
  auto axisS = std::make_unique<ValueType[]>(params.nSampNumInBin);
  process::for_each(
      params.nSampNumInBin, [&](std::size_t i) { axisS[i] = meshS[i]; }, basic::CpuMultiThread(thread_num));
  ValueType axisSStart = axisS[0];
  ValueType axisGap = axisS[1] - axisS[0];
  BackProjectionCore<ValueType> backProjectionCore{reconImg.get(), filterSino.get(), meshTheta.get(),
                                                   meshImgX.get(), meshImgY.get(),   params,
                                                   nRebinSliceNum, axisSStart,       axisGap};
  const auto totalBackProjectionPoints = nRebinSliceNum * params.nSampNumInView * params.nImgWidth * params.nImgHeight;
  auto dummyOutput4 = std::make_unique<float[]>(totalBackProjectionPoints);
  process::for_each(
      totalBackProjectionPoints,
      [&](std::size_t i) {
        backProjectionCore(i);
        dummyOutput4[i] = ValueType(0);
      },
      basic::CpuMultiThread(thread_num));

  // Step 5: 将重建图像数据复制到输出图像
  std::copy(reconImg.get(), reconImg.get() + reconImgSize, outputImage);
}
} // namespace openpni::process::fbp

// #include "../Undef.h"

// 将 saveComplexData 函数改为独立的非成员函数
namespace openpni::process::fbp {

void saveComplexData(
    const std::string &filename, const std::vector<std::vector<std::vector<std::complex<double>>>> &data) {
  std::ofstream realFile(filename + "_real.bin", std::ios::binary);
  std::ofstream imagFile(filename + "_imag.bin", std::ios::binary);

  for (const auto &plane : data) {
    for (const auto &row : plane) {
      for (const auto &value : row) {
        double realPart = value.real();
        double imagPart = value.imag();
        realFile.write(reinterpret_cast<const char *>(&realPart), sizeof(double));
        imagFile.write(reinterpret_cast<const char *>(&imagPart), sizeof(double));
      }
    }
  }

  realFile.close();
  imagFile.close();
}

} // namespace openpni::process::fbp
