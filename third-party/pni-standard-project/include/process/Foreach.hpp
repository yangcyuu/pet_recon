#pragma once
#include <algorithm>
#include <cmath>
#include <execution>
#include <functional>
#include <numbers>
#include <ranges>
#include <thread>
#include <type_traits>

#include "../PnI-Config.hpp"
#include "../basic/CpuInfo.hpp"
#include "../basic/Image.hpp"
#include "../basic/Point.hpp"
#include "../misc/Halflife.hpp"
namespace openpni::process {
template <typename T>
concept PointWiseMethod = requires(const T &method, std::size_t i) { std::is_arithmetic_v<decltype(method(i))>; };

namespace impl {
template <typename Func>
inline void for_each_static_impl(
    std::size_t __start, std::size_t __step, std::size_t __max, const Func &__func) {
  for (std::size_t i = __start; i < __max; i += __step)
    __func(i);
}
template <typename Func>
inline void for_each_dynamic_impl(
    std::atomic<std::size_t> &__index, std::size_t __max, std::size_t __scheduleSize, const Func &__func) {
  while (true) {
    const auto index = __index.fetch_add(__scheduleSize);
    if (index >= __max)
      break;
    for_each_static_impl(index, 1, std::min(index + __scheduleSize, __max), __func);
  }
}
} // namespace impl

template <typename Func>
inline void for_each(
    std::size_t __max, Func &&__func, basic::CpuMultiThread __cpuMultiThread = cpu_threads.singleThread()) {
  if (__max <= 0)
    return;
  if (__cpuMultiThread.threadNum() <= 1) {
    const auto iota = std::views::iota(std::size_t{0}, __max);
    std::for_each(std::execution::par, iota.begin(), iota.end(), std::forward<Func>(__func));
    return;
  }

  std::vector<std::jthread> threads;
  if (__cpuMultiThread.scheduleType() == basic::CpuMultiThread::ScheduleType::Static) {
    const auto realThreadNum = std::min<std::size_t>(__cpuMultiThread.threadNum(), __max);
    for (const auto threadIndex : std::views::iota(0ull, realThreadNum))
      threads.emplace_back([&, threadIndex] { impl::for_each_static_impl(threadIndex, realThreadNum, __max, __func); });
  } else {
    const auto scheduleSize = __cpuMultiThread.scheduleNum(); // >=1 is assumed
    const auto realThreadNum = basic::dev_ceil<std::size_t>(__max, scheduleSize);
    std::atomic<std::size_t> index{0};
    for (const auto threadIndex : std::views::iota(0ull, realThreadNum))
      threads.emplace_back([&, threadIndex] { impl::for_each_dynamic_impl(index, __max, scheduleSize, __func); });
  }
}

template <typename Precision, typename InputValueType>
struct LinearMapping {
  Precision slope;
  Precision intercept;
  InputValueType *input;

  __PNI_CUDA_MACRO__ auto operator()(
      std::size_t i) const {
    return slope * input[i] + intercept;
  }
};

template <typename InputValueType>
__PNI_CUDA_MACRO__ LinearMapping<double, InputValueType> decay(
    InputValueType *__input, double __secondsBeforeScan, double __secondsDuringScan, double __secondsHalfLife,
    double __sourceBranchingRatio) {
  const basic::Dose doseNow{1., 0., __secondsHalfLife};
  const auto doseScanStart = doseNow << __secondsDuringScan;
  const auto doseBeforeScan = doseScanStart << __secondsBeforeScan;

  const auto averageDoseDuringScan =
      (std::exp2(__secondsDuringScan / __secondsHalfLife) - 1) / std::numbers::ln2 * __secondsDuringScan;

  double slope;
  slope = doseBeforeScan.dose / doseScanStart.dose / __sourceBranchingRatio;
  if (__secondsDuringScan > 1e-3)
    slope *= doseScanStart.dose / averageDoseDuringScan;

  return LinearMapping<double, InputValueType>{slope, 0., __input};
}

template <typename InputValueType>
__PNI_CUDA_MACRO__ LinearMapping<double, InputValueType> well_counter(
    InputValueType *__input, double __slope) {
  return LinearMapping<double, InputValueType>{__slope, 0., __input};
}

template <FloatingPoint_c InputValueType, FloatingPoint_c OutputValueType>
struct CBCTPreprocessing {
  const InputValueType *rawProj;
  const InputValueType *airProj;
  __PNI_CUDA_MACRO__ OutputValueType operator()(
      std::size_t i) const {
    return OutputValueType(-1.) * std::log(OutputValueType(rawProj[i]) / airProj[i]);
  }
};

template <FloatingPoint_c ValueType, FloatingPoint_c OutputValueType>
struct CBCTBeamHarden {
  const ValueType *inputProj;
  __PNI_CUDA_MACRO__ ValueType operator()(
      std::size_t i) const {
    return inputProj[i] < 1e-6 ? 0.0f : powf(fabsf(expf(inputProj[i]) - 1.0f), 0.8f);
  }
};

template <FloatingPoint_c ValueType>
struct CBCTWeight {
  const ValueType *inputProj;
  openpni::basic::Vec2<unsigned> crystalNum;
  unsigned angleNum;
  openpni::basic::Vec2<float> crystalSize;
  openpni::basic::Vec2<float> center;
  ValueType SDD;
  __PNI_CUDA_MACRO__ ValueType operator()(
      std::size_t i) const {
    openpni::basic::Vec2<unsigned> indexInProjection =
        basic::make_vec2<unsigned>((i / angleNum) % crystalNum.x, (i / angleNum) / crystalNum.x);
    return inputProj[i] * (SDD / sqrt(((indexInProjection + 0.5) * crystalSize - center).l2() + SDD * SDD));
  }
};

template <FloatingPoint_c ValueType, FloatingPoint_c InputValueType>
struct CBCTPostProcess {
  const ValueType fCTSlope;
  const ValueType fCTIntercept;
  const basic::Vec3<unsigned> voxelNum;
  InputValueType *input;
  __PNI_CUDA_MACRO__ InputValueType operator()(
      std::size_t i) const {
    InputValueType value = input[i] * fCTSlope + fCTIntercept;
    value = value > -1000.0f ? value : -1000.0f;

    const unsigned sliceSize = voxelNum.x * voxelNum.y;
    // const unsigned idxInSlice = i % sliceSize;
    const basic::Vec3<unsigned> idx3d{static_cast<unsigned>(i % sliceSize % voxelNum.x),
                                      static_cast<unsigned>((i % sliceSize) / voxelNum.x),
                                      static_cast<unsigned>(i / sliceSize)};

    // value = (idx3d.z == 0) ? input[idxInSlice + sliceSize] : value;
    // value = (idx3d.z == voxelNum.z - 1) ? input[idxInSlice + sliceSize * (voxelNum.z -
    // 2)] : value;
    const basic::Vec2<float> center{voxelNum.x * 0.5f, voxelNum.y * 0.5f};
    const basic::Vec2<float> pos{idx3d.x + 0.5f, idx3d.y + 0.5f};
    const float dist2 = (pos - center).l2();
    const float rr = voxelNum.y * 0.5f - 8.0f;
    value = (dist2 > rr * rr) ? -1000.0f : value;
    return value;
  }
};

template <FloatingPoint_c ValueType, FloatingPoint_c InputValueType>
struct CBCTCO {
  const ValueType offsetX;
  const ValueType offsetY;
  const basic::Vec3<unsigned> voxelNum;
  const basic::Vec3<float> voxelSize;
  const basic::Vec3<float> voxelCenter;
  InputValueType *input;
  __PNI_CUDA_MACRO__ InputValueType operator()(
      std::size_t i) const {
    const unsigned sliceSize = voxelNum.x * voxelNum.y;
    const unsigned zIndex = i / sliceSize; // 当前体素所在的层
    const unsigned idxInSlice = i % sliceSize;
    const unsigned yIndex = idxInSlice / voxelNum.x;
    const unsigned xIndex = idxInSlice % voxelNum.x;
    // 计算当前体素的物理坐标
    ValueType xpos = ((ValueType)xIndex - (ValueType)voxelNum.x / 2 + 0.5) * voxelSize.x + voxelCenter.x;
    ValueType ypos = ((ValueType)yIndex - (ValueType)voxelNum.y / 2 + 0.5) * voxelSize.y + voxelCenter.y;
    ValueType zpos = ((ValueType)zIndex - (ValueType)voxelNum.z / 2 + 0.5) * voxelSize.z + voxelCenter.z;

    // 应用偏移校正
    xpos -= offsetX;
    ypos -= offsetY;
    return GetPositionValue(xpos, ypos, zpos, true);
  }

  __PNI_CUDA_MACRO__ InputValueType GetPositionValue(
      ValueType xpos, ValueType ypos, ValueType zpos, bool linearInterpolation) const {
    xpos -= voxelCenter.x + 0.5 * voxelSize.x;
    ypos -= voxelCenter.y + 0.5 * voxelSize.y;
    zpos -= voxelCenter.z + 0.5 * voxelSize.z;

    float x = float((xpos + voxelNum.x * 0.5 * voxelSize.x) / voxelSize.x);
    int xlow = int(x);
    if (xlow < 0 || xlow >= voxelNum.x)
      return -1000.0;

    float y = float((ypos + voxelNum.y * 0.5 * voxelSize.y) / voxelSize.y);
    int ylow = int(y);
    if (ylow < 0 || ylow >= voxelNum.y)
      return -1000.0;

    float z = float((zpos + voxelNum.z * 0.5 * voxelSize.z) / voxelSize.z);
    int zlow = int(z);
    if (zlow < 0 || zlow >= voxelNum.z)
      return -1000.0;

    if (linearInterpolation) {
      int xhigh = xlow + 1;
      int yhigh = ylow + 1;
      int zhigh = zlow + 1;
      float xhw = x - float(xlow); // normalized weight of high voxel
      float yhw = y - float(ylow);
      float zhw = z - float(zlow);
      float xlw = float(xhigh) - x; // normalized weight of low voxel
      float ylw = float(yhigh) - y;
      float zlw = float(zhigh) - z;
      // To prevent crossing the image boundary
      xhigh = std::min(xhigh, int(voxelNum.x - 1));
      yhigh = std::min(yhigh, int(voxelNum.y - 1));
      zhigh = std::min(zhigh, int(voxelNum.z - 1));

      return input[xlow + ylow * voxelNum.x + zlow * voxelNum.x * voxelNum.y] * xlw * ylw * zlw +
             input[xhigh + ylow * voxelNum.x + zlow * voxelNum.x * voxelNum.y] * xhw * ylw * zlw +
             input[xlow + yhigh * voxelNum.x + zlow * voxelNum.x * voxelNum.y] * xlw * yhw * zlw +
             input[xlow + ylow * voxelNum.x + zhigh * voxelNum.x * voxelNum.y] * xlw * ylw * zhw +
             input[xhigh + yhigh * voxelNum.x + zlow * voxelNum.x * voxelNum.y] * xhw * yhw * zlw +
             input[xhigh + ylow * voxelNum.x + zhigh * voxelNum.x * voxelNum.y] * xhw * ylw * zhw +
             input[xlow + yhigh * voxelNum.x + zhigh * voxelNum.x * voxelNum.y] * xlw * yhw * zhw +
             input[xhigh + yhigh * voxelNum.x + zhigh * voxelNum.x * voxelNum.y] * xhw * yhw * zhw;
    } else {
      return input[xlow + ylow * voxelNum.x + zlow * voxelNum.x * voxelNum.y];
    }
  }
};

template <FloatingPoint_c ValueType>
struct CBCTMergeImg {
  const std::vector<std::unique_ptr<ValueType[]>> &__imgOutList;
  unsigned imgWidth;
  unsigned imgHeight;
  unsigned imgDepth;
  unsigned bedNum;
  unsigned imgOverlapSliceN;
  unsigned mergedImgDepth;
  ValueType operator()(
      std::size_t i) const {
    unsigned sliceSize = imgWidth * imgHeight;
    unsigned z = i / sliceSize;
    unsigned idxInSlice = i % sliceSize;
    unsigned group = imgDepth - imgOverlapSliceN;
    unsigned bedIdx = z / group;
    if (bedIdx >= bedNum)
      bedIdx = bedNum - 1;
    unsigned startZ = bedIdx * group;
    unsigned zInBed = z - startZ;

    if (bedIdx > 0 && zInBed < imgOverlapSliceN) {
      float ratio1 = 1.0f * zInBed / imgOverlapSliceN;
      float ratio2 = 1.0f - ratio1;
      float v1 = __imgOutList[bedIdx - 1][(imgDepth - imgOverlapSliceN + zInBed) * sliceSize + idxInSlice];
      float v2 = __imgOutList[bedIdx][zInBed * sliceSize + idxInSlice];
      return v1 * ratio2 + v2 * ratio1;
    }
    return __imgOutList[bedIdx][zInBed * sliceSize + idxInSlice];
  }
};

template <typename ValueType>
struct CBCTFlipZ {
  const ValueType *mergedImgBuffer;
  unsigned imgWidth;
  unsigned imgHeight;
  unsigned mergedImgDepth;

  ValueType operator()(
      std::size_t i) const {
    unsigned sliceSize = imgWidth * imgHeight;
    unsigned z = i / sliceSize;
    unsigned idxInSlice = i % sliceSize;
    unsigned flippedZ = mergedImgDepth - 1 - z;
    return mergedImgBuffer[flippedZ * sliceSize + idxInSlice];
  }
};

template <typename Iterator>
inline void set_max_value_to_1(
    Iterator __begin, Iterator __end) {
  using ValueType = typename std::iterator_traits<Iterator>::value_type;
  if (__begin == __end)
    return; // 空范围，直接返回

  ValueType max_value = *std::max_element(__begin, __end);
  if (max_value == 0)
    return; // 最大值为0，避免除以零

  std::for_each(__begin, __end, [max_value](ValueType &value) { value /= max_value; });
}

template <typename Range>
inline void set_max_value_to_1(
    Range &&__range) {
  if (std::ranges::size(__range) == 0)
    return; // 空范围，直接返回

  using ValueType = typename std::decay_t<decltype(*std::ranges::begin(__range))>;
  ValueType max_value = *std::ranges::max_element(__range);
  if (max_value == 0)
    return; // 最大值为0，避免除以零

  std::ranges::for_each(__range, [max_value](ValueType &value) { value /= max_value; });
}

} // namespace openpni::process
