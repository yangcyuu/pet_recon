#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <stack>
#include <string>
#include <vector>

#include "../basic/Image.hpp"
#include "../basic/Math.hpp"
#include "../basic/Point.hpp"
#include "Foreach.hpp"

namespace openpni::process {
// 几何校正特有的参数结构
struct GeometryCorrectionParams {
  int ballNum;             // 钢珠数量
  double ballDis;          // 钢珠间距 (mm)
  double thresholdPercent; // 阈值百分比
};

// 几何校正结果
struct GeometryCorrectionResult {
  float SOD;           // Source to Object Distance
  float SDD;           // Source to Detector Distance
  float offsetU;       // U方向偏移
  float offsetV;       // V方向偏移
  float rotationAngle; // 旋转角度
  bool success;        // 校正是否成功
};

// 椭圆参数结构
struct Ellipse {
  openpni::p2df coordA; // 长轴端点（大）
  openpni::p2df coordB; // 长轴端点（小）
  openpni::p2df coordC; // 短轴端点（小）
  openpni::p2df coordD; // 短轴端点（大）
  openpni::p2df coordO; // 中心点
};

// 几何校正主结构体
struct _GeometryCorrection {

  // 投影阈值处理结构体
  template <typename CalculatePrecision = float>
  struct thresholding {
    const CalculatePrecision *projections_in;
    int *segmented_out;
    int projSizeU, projSizeV;
    int projectionNum;
    CalculatePrecision thresholdRatio;

    thresholding(
        const CalculatePrecision *__projections_in, int *__segmented_out, int __projSizeU, int __projSizeV,
        int __projectionNum, CalculatePrecision __thresholdRatio)
        : projections_in(__projections_in)
        , segmented_out(__segmented_out)
        , projSizeU(__projSizeU)
        , projSizeV(__projSizeV)
        , projectionNum(__projectionNum)
        , thresholdRatio(__thresholdRatio) {}

    void operator()(
        size_t projectionIdx) const {
      if (projectionIdx >= static_cast<size_t>(projectionNum))
        return;

      int totalSize = projSizeU * projSizeV;
      auto projPtr = &projections_in[projectionIdx * totalSize];
      auto segPtr = &segmented_out[projectionIdx * totalSize];

      // 1. 找到该投影的最大值
      CalculatePrecision maxValue = -1.0f;
      for (int i = 0; i < totalSize; i++) {
        maxValue = (projPtr[i] > maxValue ? projPtr[i] : maxValue);
      }

      // 2. 二值化处理：小于阈值的设为1，其他设为0
      CalculatePrecision threshold = maxValue * thresholdRatio;
      for (int i = 0; i < totalSize; i++) {
        segPtr[i] = (projPtr[i] < threshold ? 1 : 0);
      }
    }
  };

  // 连通域标记结构体（Two-pass算法）
  template <typename CalculatePrecision = float>
  struct connected_component_labeling {
    int *segmented_inout;
    int projSizeU, projSizeV;
    int projectionNum;
    int expectedBallNum;

    connected_component_labeling(
        int *__segmented_inout, int __projSizeU, int __projSizeV, int __projectionNum, int __expectedBallNum)
        : segmented_inout(__segmented_inout)
        , projSizeU(__projSizeU)
        , projSizeV(__projSizeV)
        , projectionNum(__projectionNum)
        , expectedBallNum(__expectedBallNum) {}

    void operator()(
        size_t projectionIdx) const {
      if (projectionIdx >= static_cast<size_t>(projectionNum))
        return;

      int totalSize = projSizeU * projSizeV;
      auto segPtr = &segmented_inout[projectionIdx * totalSize];

      // 1. 将前景像素标记为0，背景像素标记为-1
      for (int i = 0; i < totalSize; i++) {
        segPtr[i] -= 1; // 原来1变成0（前景），原来0变成-1（背景）
      }

      // Two-pass连通域标记算法
      int maxLabelIndex = 1;
      std::vector<int> labelParent;
      labelParent.push_back(0);

      // Pass 1: 初始标记
      for (int j = 0; j < projSizeV; j++) {
        for (int i = 0; i < projSizeU; i++) {
          int idx = i + j * projSizeU;

          if (segPtr[idx] == -1) { // 背景像素
            continue;
          }

          bool leftFore = (i - 1 >= 0 && segPtr[i - 1 + j * projSizeU] >= 0);
          bool topFore = (j - 1 >= 0 && segPtr[i + (j - 1) * projSizeU] >= 0);

          if (!leftFore && !topFore) {
            // 新的连通域
            segPtr[idx] = maxLabelIndex;
            labelParent.push_back(0);
            maxLabelIndex++;
          } else if (leftFore && !topFore) {
            // 继承左邻像素的标签
            segPtr[idx] = segPtr[i - 1 + j * projSizeU];
          } else if (topFore && !leftFore) {
            // 继承上邻像素的标签
            segPtr[idx] = segPtr[i + (j - 1) * projSizeU];
          } else if (leftFore && topFore) {
            // 两个邻居都是前景，处理标签等价关系
            int leftLabel = segPtr[i - 1 + j * projSizeU];
            int topLabel = segPtr[i + (j - 1) * projSizeU];
            int minLabel = std::min(leftLabel, topLabel);
            int maxLabel = std::max(leftLabel, topLabel);

            segPtr[idx] = minLabel;

            if (minLabel != maxLabel) {
              if (labelParent[maxLabel] == 0) {
                labelParent[maxLabel] = minLabel;
              } else {
                labelParent[maxLabel] = std::min(minLabel, labelParent[maxLabel]);
              }
            }
          }
        }
      }

      // Pass 2: 合并等价标签
      for (size_t i = 1; i < labelParent.size(); i++) {
        int labelId = i;
        int parentLabelId = labelParent[i];

        if (parentLabelId == 0) {
          continue;
        }

        while (parentLabelId != 0) {
          labelId = parentLabelId;
          parentLabelId = labelParent[labelId];
        }

        labelParent[i] = labelId;
      }

      // 收集根标签
      std::vector<int> rootLabels;
      for (size_t i = 1; i < labelParent.size(); i++) {
        if (labelParent[i] == 0) {
          rootLabels.push_back(i);
        }
      }

      // 检查连通域数量是否符合预期
      if (static_cast<int>(rootLabels.size()) != expectedBallNum) {
        // 如果连通域数量不符合预期，标记为失败（全部设为0）
        for (int i = 0; i < totalSize; i++) {
          segPtr[i] = 0;
        }
        return;
      }

      // 重新标记连通域
      for (int i = 0; i < totalSize; i++) {
        if (segPtr[i] == -1) {
          segPtr[i] = 0; // 背景
        } else if (labelParent[segPtr[i]] != 0) {
          segPtr[i] = labelParent[segPtr[i]];
        }
      }

      // 将标签重新映射为1, 2, 3, ...
      for (int i = 0; i < totalSize; i++) {
        if (segPtr[i] != 0) {
          auto it = std::find(rootLabels.begin(), rootLabels.end(), segPtr[i]);
          if (it != rootLabels.end()) {
            segPtr[i] = static_cast<int>(std::distance(rootLabels.begin(), it) + 1);
          }
        }
      }
    }
  };

  // 边界处理结构体
  template <typename CalculatePrecision = float>
  struct boundary_processing {
    CalculatePrecision *projections_inout;
    int projSizeU, projSizeV;
    int projectionNum;
    int boundaryWidth;

    boundary_processing(
        CalculatePrecision *__projections_inout, int __projSizeU, int __projSizeV, int __projectionNum,
        int __boundaryWidth = 5)
        : projections_inout(__projections_inout)
        , projSizeU(__projSizeU)
        , projSizeV(__projSizeV)
        , projectionNum(__projectionNum)
        , boundaryWidth(__boundaryWidth) {}

    void operator()(
        size_t projectionIdx) const {
      if (projectionIdx >= static_cast<size_t>(projectionNum))
        return;

      auto projPtr = &projections_inout[projectionIdx * projSizeU * projSizeV];

      // 获取替换值（从内部采样点）
      CalculatePrecision replaceValue = projPtr[boundaryWidth * projSizeU + boundaryWidth];

      // 处理边界像素
      for (int j = 0; j < projSizeV; j++) {
        for (int i = 0; i < projSizeU; i++) {
          // 如果是边界像素（不在内部区域）
          if (!(j >= boundaryWidth && j < projSizeV - boundaryWidth && i >= boundaryWidth &&
                i < projSizeU - boundaryWidth)) {
            projPtr[i + j * projSizeU] = replaceValue;
          }
        }
      }
    }
  };

  // 均值滤波结构体
  template <typename CalculatePrecision = float>
  struct mean_filter {
    CalculatePrecision *projections_inout;
    int projSizeU, projSizeV;
    int projectionNum;
    int windowSize;
    int boundarySkip;

    mean_filter(
        CalculatePrecision *__projections_inout, int __projSizeU, int __projSizeV, int __projectionNum,
        int __windowSize = 5, int __boundarySkip = 5)
        : projections_inout(__projections_inout)
        , projSizeU(__projSizeU)
        , projSizeV(__projSizeV)
        , projectionNum(__projectionNum)
        , windowSize(__windowSize)
        , boundarySkip(__boundarySkip) {}

    void operator()(
        size_t projectionIdx) const {
      if (projectionIdx >= static_cast<size_t>(projectionNum))
        return;

      auto projPtr = &projections_inout[projectionIdx * projSizeU * projSizeV];

      // 创建临时缓冲区存储滤波结果
      auto tempBuffer = std::make_unique<CalculatePrecision[]>(projSizeU * projSizeV);
      std::copy(projPtr, projPtr + projSizeU * projSizeV, tempBuffer.get());

      for (int j = 0; j < projSizeV; j++) {
        for (int i = 0; i < projSizeU; i++) {
          // 跳过边界像素（使用参数化的边界宽度）
          if (i < boundarySkip || j < boundarySkip || i >= projSizeU - boundarySkip || j >= projSizeV - boundarySkip) {
            continue;
          }

          CalculatePrecision mean = 0.0f;
          int halfWin = windowSize / 2;

          // 计算窗口内像素的均值
          for (int jj = std::max(j - halfWin, 0); jj <= std::min(j + halfWin, projSizeV - 1); jj++) {
            for (int ii = std::max(i - halfWin, 0); ii <= std::min(i + halfWin, projSizeU - 1); ii++) {
              mean += tempBuffer[ii + jj * projSizeU];
            }
          }

          projPtr[i + j * projSizeU] = mean / (windowSize * windowSize);
        }
      }

      // 裁剪边界：裁剪掉四个边界各1/7的区域，设置为图像最大值
      int cropU = projSizeU / 7; // U方向裁剪宽度
      int cropV = projSizeV / 7; // V方向裁剪宽度

      // 查找滤波后图像的最大值
      CalculatePrecision maxValue = -1.0f;
      for (int i = 0; i < projSizeU * projSizeV; i++) {
        maxValue = (projPtr[i] > maxValue ? projPtr[i] : maxValue);
      }

      // 将裁剪区域设置为最大值
      for (int j = 0; j < projSizeV; j++) {
        for (int i = 0; i < projSizeU; i++) {
          // 检查是否在裁剪区域内（上下左右各1/7）
          if (i < cropU || i >= projSizeU - cropU || j < cropV || j >= projSizeV - cropV) {
            projPtr[i + j * projSizeU] = maxValue;
          }
        }
      }
    }
  };

  // 钢珠检测结构体（基于分割数据和原始数据的加权计算）
  template <typename CalculatePrecision = float>
  struct ball_detection {
    const int *segmented_in;
    const CalculatePrecision *raw_projections_in;
    openpni::p2df *ball_centers_out;
    int projSizeU, projSizeV;
    int projectionNum, ballNum;
    openpni::p2df pixelSize;

    ball_detection(
        const int *__segmented_in, const CalculatePrecision *__raw_projections_in, openpni::p2df *__ball_centers_out,
        int __projSizeU, int __projSizeV, int __projectionNum, int __ballNum, openpni::p2df __pixelSize)
        : segmented_in(__segmented_in)
        , raw_projections_in(__raw_projections_in)
        , ball_centers_out(__ball_centers_out)
        , projSizeU(__projSizeU)
        , projSizeV(__projSizeV)
        , projectionNum(__projectionNum)
        , ballNum(__ballNum)
        , pixelSize(__pixelSize) {}

    void operator()(
        size_t projectionIdx) const {
      if (projectionIdx >= static_cast<size_t>(projectionNum))
        return;

      int totalSize = projSizeU * projSizeV;
      auto segPtr = &segmented_in[projectionIdx * totalSize];
      auto rawPtr = &raw_projections_in[projectionIdx * totalSize];

      // 为每个钢珠计算加权中心
      std::vector<CalculatePrecision> coordSumX(ballNum, 0);
      std::vector<CalculatePrecision> coordSumY(ballNum, 0);
      std::vector<CalculatePrecision> pixelWeight(ballNum, 0);

      for (int j = 0; j < projSizeV; j++) {
        for (int i = 0; i < projSizeU; i++) {
          int idx = i + j * projSizeU;
          int ballIndex = segPtr[idx];

          // 如果像素属于某个钢珠区域 (1到ballNum)
          if (ballIndex >= 1 && ballIndex <= ballNum) {
            CalculatePrecision intensity = rawPtr[idx];
            int ballIdx = ballIndex - 1;

            // 累加坐标*强度和强度
            coordSumX[ballIdx] += i * intensity;
            coordSumY[ballIdx] += j * intensity;
            pixelWeight[ballIdx] += intensity;
          }
        }
      }

      // 计算每个钢珠的中心并转换为物理坐标
      for (int ballIdx = 0; ballIdx < ballNum; ++ballIdx) {
        if (pixelWeight[ballIdx] > 0) {
          CalculatePrecision centerX = coordSumX[ballIdx] / pixelWeight[ballIdx];
          CalculatePrecision centerY = coordSumY[ballIdx] / pixelWeight[ballIdx];

          // 按照UCorrCTGeometry的坐标转换方式和存储方式
          ball_centers_out[ballIdx * projectionNum + projectionIdx] =
              openpni::p2df{static_cast<float>(centerX * pixelSize.x + pixelSize.x / 2),
                            static_cast<float>(centerY * pixelSize.y + pixelSize.y / 2)};
        } else {
          ball_centers_out[ballIdx * projectionNum + projectionIdx] = openpni::p2df{-1.0f, -1.0f};
        }
      }
    }
  };

  // 椭圆拟合结构体（基于UCorrCTGeometry::CalcEllipsePara）
  template <typename CalculatePrecision = float>
  struct ellipse_fitting {
    const openpni::p2df *ball_centers_in;
    Ellipse *ellipse_out;
    int projectionNum, ballNum;

    ellipse_fitting(
        const openpni::p2df *__ball_centers_in, Ellipse *__ellipse_out, int __projectionNum, int __ballNum)
        : ball_centers_in(__ball_centers_in)
        , ellipse_out(__ellipse_out)
        , projectionNum(__projectionNum)
        , ballNum(__ballNum) {}

    void operator()(
        size_t ballIdx) const {
      if (ballIdx >= static_cast<size_t>(ballNum))
        return;

      // 收集该钢珠在所有投影中的轨迹点 - 按照UCorrCTGeometry的数据排列
      std::vector<openpni::p2df> trajectory;
      trajectory.reserve(projectionNum);

      for (int projIdx = 0; projIdx < projectionNum; ++projIdx) {
        trajectory.push_back(ball_centers_in[ballIdx * projectionNum + projIdx]);
      }

      // 按照UCorrCTGeometry::CalcEllipsePara实现椭圆参数计算
      int halfNum = projectionNum / 2;
      CalculatePrecision maxDis = -1.0f, minDis = 1000.0f;
      int indexMax = 0, indexMin = 0;

      // 寻找最大距离和最小距离的点对（相隔180度）
      for (int i = 0; i < halfNum; i++) {
        int j = i + halfNum;
        if (j >= projectionNum)
          continue;

        // 内联距离计算
        CalculatePrecision dx = trajectory[i].x - trajectory[j].x;
        CalculatePrecision dy = trajectory[i].y - trajectory[j].y;
        CalculatePrecision dis = sqrt(dx * dx + dy * dy);

        if (dis > maxDis) {
          maxDis = dis;
          indexMax = i;
        }

        if (dis < minDis) {
          minDis = dis;
          indexMin = i;
        }
      }

      // 设置椭圆的长轴和短轴端点
      ellipse_out[ballIdx].coordA = trajectory[indexMax];           // 长轴端点A
      ellipse_out[ballIdx].coordB = trajectory[indexMax + halfNum]; // 长轴端点B
      ellipse_out[ballIdx].coordC = trajectory[indexMin];           // 短轴端点C
      ellipse_out[ballIdx].coordD =
          trajectory[indexMin + halfNum]; // 短轴端点D      // 计算椭圆中心（所有轨迹点的平均）

      openpni::p2df center = {0.0f, 0.0f};
      for (const auto &point : trajectory) {
        center.x += point.x;
        center.y += point.y;
      }
      center.x /= trajectory.size();
      center.y /= trajectory.size();

      ellipse_out[ballIdx].coordO = center;
    }
  };

  // 几何参数计算结构体
  template <typename CalculatePrecision = float>
  struct geometry_calculation {
    const Ellipse *ellipses_in;
    GeometryCorrectionResult *result_out;
    int ballNum;
    CalculatePrecision ballDis;
    openpni::p2df pixelSize;
    int projSizeU, projSizeV;

    geometry_calculation(
        const Ellipse *__ellipses_in, GeometryCorrectionResult *__result_out, int __ballNum,
        CalculatePrecision __ballDis, openpni::p2df __pixelSize, int __projSizeU, int __projSizeV)
        : ellipses_in(__ellipses_in)
        , result_out(__result_out)
        , ballNum(__ballNum)
        , ballDis(__ballDis)
        , pixelSize(__pixelSize)
        , projSizeU(__projSizeU)
        , projSizeV(__projSizeV) {}

    // 线性拟合函数
    bool linearFit(
        CalculatePrecision &k, CalculatePrecision &b, const CalculatePrecision *X, const CalculatePrecision *Y,
        int N) const {
      CalculatePrecision SumX = 0, SumY = 0, SumXY = 0, SumXSquare = 0;

      for (int i = 0; i < N; i++) {
        SumX += X[i];
        SumY += Y[i];
        SumXY += X[i] * Y[i];
        SumXSquare += X[i] * X[i];
      }

      k = (SumX * SumY - N * SumXY) / (SumX * SumX - N * SumXSquare);
      b = SumY / N - k * SumX / N;

      return true;
    }

    void operator()() const {
      // 按照UCorrCTGeometry.cpp中的CalcGeometryParas算法实现

      // // 首先打印椭圆参数（对应原版的椭圆参数输出）
      // for (int i = 0; i < ballNum; i++) {
      //   printf("E%d. A:%f,%f; B:%f,%f; C:%f,%f; D:%f,%f; O:%f,%f\n", i + 1, ellipses_in[i].coordA.x,
      //          ellipses_in[i].coordA.y, ellipses_in[i].coordB.x, ellipses_in[i].coordB.y, ellipses_in[i].coordC.x,
      //          ellipses_in[i].coordC.y, ellipses_in[i].coordD.x, ellipses_in[i].coordD.y, ellipses_in[i].coordO.x,
      //          ellipses_in[i].coordO.y);
      // }

      std::vector<CalculatePrecision> coordX, coordY;
      std::vector<CalculatePrecision> coordO_U, coordO_V;

      // 计算SDD和OffsetV - 使用椭圆短轴和长轴比例
      for (int i = 0; i < ballNum; i++) {
        // 椭圆短轴中点的V坐标
        CalculatePrecision midY = (ellipses_in[i].coordC.y + ellipses_in[i].coordD.y) / 2;
        // 按照原版UCorrCTGeometry: (CoordC.y - CoordD.y) / 长轴距离
        CalculatePrecision longAxisDx = ellipses_in[i].coordA.x - ellipses_in[i].coordB.x;
        CalculatePrecision longAxisDy = ellipses_in[i].coordA.y - ellipses_in[i].coordB.y;
        CalculatePrecision longAxisDistance = sqrt(longAxisDx * longAxisDx + longAxisDy * longAxisDy);
        CalculatePrecision axisRatio = (ellipses_in[i].coordC.y - ellipses_in[i].coordD.y) / longAxisDistance;

        coordX.push_back(midY);
        coordY.push_back(axisRatio);
      }

      CalculatePrecision K1, B1;
      linearFit(K1, B1, coordY.data(), coordX.data(), ballNum);

      CalculatePrecision offsetV = B1;
      CalculatePrecision SDD = abs(K1);

      // 计算OffsetU和旋转角度 - 使用椭圆中心点的线性拟合
      for (int i = 0; i < ballNum; i++) {
        coordO_U.push_back(ellipses_in[i].coordO.x);
        coordO_V.push_back(ellipses_in[i].coordO.y);
      }

      CalculatePrecision K2, B2;
      linearFit(K2, B2, coordO_V.data(), coordO_U.data(), ballNum);
      CalculatePrecision rotationAngle = atan(K2);
      CalculatePrecision offsetU = K2 * offsetV + B2;

      // 计算SOD - 根据钢珠间距和椭圆中心距离的比例 - 内联距离计算
      CalculatePrecision centerDx = ellipses_in[0].coordO.x - ellipses_in[ballNum - 1].coordO.x;
      CalculatePrecision centerDy = ellipses_in[0].coordO.y - ellipses_in[ballNum - 1].coordO.y;
      CalculatePrecision centerDistance = sqrt(centerDx * centerDx + centerDy * centerDy);
      CalculatePrecision SOD = SDD * ballDis * (ballNum - 1) / centerDistance;

      // 转换到探测器中心坐标系
      offsetU -= projSizeU * pixelSize.x / 2;
      offsetV -= projSizeV * pixelSize.y / 2;

      // 设置结果
      result_out->SOD = SOD;
      result_out->SDD = SDD;
      result_out->offsetU = offsetU;
      result_out->offsetV = offsetV;
      result_out->rotationAngle = rotationAngle;
      result_out->success = true;
    }
  };

public:
  // 主处理函数
  template <typename ProjectionValueType>
  GeometryCorrectionResult operator()(
      ProjectionValueType const *const *projectionDataPtrs, int projectionNum, int projSizeU, int projSizeV,
      openpni::p2df pixelSize, const GeometryCorrectionParams &params) const {

    using CalculatePrecision = float;
    GeometryCorrectionResult result;

    const auto totalPixelNum = projSizeU * projSizeV * projectionNum;

    // 分配工作内存
    auto raw_projections = std::make_unique<CalculatePrecision[]>(totalPixelNum);
    auto segmented_projections = std::make_unique<int[]>(totalPixelNum);
    auto ball_centers = std::make_unique<openpni::p2df[]>(projectionNum * params.ballNum);
    auto ellipses = std::make_unique<Ellipse[]>(params.ballNum);

    // 步骤1: 数据复制
    for (int projIdx = 0; projIdx < projectionNum; ++projIdx) {
      auto src = projectionDataPtrs[projIdx];
      auto dst = &raw_projections[projIdx * projSizeU * projSizeV];
      for (int i = 0; i < projSizeU * projSizeV; ++i) {
        dst[i] = static_cast<CalculatePrecision>(src[i]);
      }
    }

    // 步骤2: 边界处理
    boundary_processing<CalculatePrecision> boundary_proc(raw_projections.get(), projSizeU, projSizeV, projectionNum,
                                                          5); // 固定边界宽度为5像素
    process::for_each(projectionNum, boundary_proc, openpni::cpu_threads.multiThreads());

    // 步骤3: 均值滤波
    mean_filter<CalculatePrecision> mean_filt(raw_projections.get(), projSizeU, projSizeV, projectionNum, 5,
                                              5); // 固定滤波窗口5x5，边界跳过5像素
    process::for_each(projectionNum, mean_filt, openpni::cpu_threads.multiThreads());

    // 步骤4: 阈值分割
    thresholding<CalculatePrecision> thresh(raw_projections.get(), segmented_projections.get(), projSizeU, projSizeV,
                                            projectionNum, static_cast<CalculatePrecision>(params.thresholdPercent));
    process::for_each(projectionNum, thresh, openpni::cpu_threads.multiThreads());

    // 步骤5: 连通域标记
    connected_component_labeling<CalculatePrecision> ccl(segmented_projections.get(), projSizeU, projSizeV,
                                                         projectionNum, params.ballNum);
    process::for_each(projectionNum, ccl, openpni::cpu_threads.multiThreads());

    // 步骤6: 钢珠检测（基于分割数据和原始数据）
    ball_detection<CalculatePrecision> detector(segmented_projections.get(), raw_projections.get(), ball_centers.get(),
                                                projSizeU, projSizeV, projectionNum, params.ballNum, pixelSize);

    // 使用process::for_each处理钢珠检测
    process::for_each(projectionNum, detector, openpni::cpu_threads.multiThreads());

    // 步骤7: 椭圆拟合
    ellipse_fitting<CalculatePrecision> fitter(ball_centers.get(), ellipses.get(), projectionNum, params.ballNum);

    // 使用process::for_each处理椭圆拟合
    process::for_each(params.ballNum, fitter, openpni::cpu_threads.multiThreads());

    // 步骤8: 几何参数计算
    geometry_calculation<CalculatePrecision> calculator(ellipses.get(), &result, params.ballNum,
                                                        static_cast<CalculatePrecision>(params.ballDis), pixelSize,
                                                        projSizeU, projSizeV);

    calculator();

    return result;
  }
};

// 全局常量实例
inline constexpr _GeometryCorrection GeometryCorrection{};

} // namespace openpni::process
