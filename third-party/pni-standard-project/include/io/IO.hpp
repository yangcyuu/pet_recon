#pragma once
#include "ListmodeIO.hpp"
#include "RawDataIO.hpp"
#include "RawImageIO.hpp"
#include "SingleIO.hpp"
namespace openpni::io {
/**
 * 段式文件格式总体规则：
 * 1. 开头若干定长字节为文件头，标识的类型、版本、段数等信息
 * 2. 文件头后分为若干段，每段包含一个段头和若干数据
 * 3. 有或没有文件尾
 *
 * 段式通常会如下实现：
 * 1. 在open阶段查看文件是否满足格式等要求，如果不满足则抛出异常
 * 2. 其他函数一般不会抛出异常
 * 3. 存取函数一般是异步的：
 *   - 存储函数依靠内部线程实际写入
 *   - 读取函数提供预取与命中机制
 *   -
 * 不保证线程安全，用户调用时需要自行保证线程安全(如仅使用一个线程调用文件对象)
 */
using RawFileInput = rawdata::RawFileInput;              // 用于将RawData文件输入程序的类
using RawFileOutput = rawdata::RawFileOutput;            // 用于将RawData输出文件的类
using SingleFileInput = single::SingleFileInput;         // 用于将单个文件输入程序的类
using SingleFileOutput = single::SingleFileOutput;       // 用于将单个文件输出程序的类
using ListmodeFileInput = listmode::ListmodeFileInput;   // 用于将Listmode文件输入程序的类
using ListmodeFileOutput = listmode::ListmodeFileOutput; // 用于将Listmode文件输入程序的类

using U16Image = rawimage::Raw3D<uint16_t>; // uint16_t  image
using U32Image = rawimage::Raw3D<uint32_t>; // uint32_t  image
using F32Image = rawimage::Raw3D<float>;    // float  image
using F64Image = rawimage::Raw3D<double>;   // double  image

template <typename T>
concept SegmentLikeFileInput =
    std::is_same_v<T, RawFileInput> || std::is_same_v<T, SingleFileInput> || std::is_same_v<T, ListmodeFileInput>;

struct _SegmentView {
  uint32_t segmentIndex;
  uint64_t dataIndexBegin;
  uint64_t dataIndexEnd;
};

template <SegmentLikeFileInput T>
inline auto selectSegments(
    T &fileInput, uint64_t clockBegin = 0, uint64_t clockEnd = uint64_t(-1)) noexcept -> std::vector<_SegmentView> {
  std::vector<_SegmentView> segments;
  const auto segmentCount = fileInput.segmentNum();
  for (uint32_t i = 0; i < segmentCount; ++i) {
    const auto header = fileInput.segmentHeader(i);
    const uint64_t segmentClockBegin = header.clock;
    const uint64_t segmentClockEnd = header.clock + header.duration;
    if (clockEnd < segmentClockBegin || clockBegin > segmentClockEnd)
      continue; // 不在范围内
    _SegmentView view;
    view.segmentIndex = i;
    if (clockBegin > segmentClockBegin && clockEnd > segmentClockEnd) {
      view.dataIndexBegin = header.count * (1 - double(clockBegin - segmentClockBegin) / double(header.duration));
      view.dataIndexEnd = header.count;
    } else if (clockBegin <= segmentClockBegin && clockEnd >= segmentClockEnd) {
      view.dataIndexBegin = 0;
      view.dataIndexEnd = header.count;
    } else if (clockEnd > segmentClockBegin && clockEnd < segmentClockEnd) {
      view.dataIndexBegin = 0;
      view.dataIndexEnd = header.count * (double(clockEnd - segmentClockBegin) / double(header.duration));
    } else
    //(clockBegin > segmentClockBegin && clockEnd < segmentClockEnd)
    {
      view.dataIndexBegin = header.count * (double(clockBegin - segmentClockBegin) / double(header.duration));
      view.dataIndexEnd = header.count * (double(clockEnd - segmentClockBegin) / double(header.duration));
    }
    view.dataIndexEnd = std::max(view.dataIndexEnd, header.count);
    view.dataIndexBegin = std::min(view.dataIndexBegin, view.dataIndexEnd);

    segments.push_back(view);
  }
  return segments;
}

}; // namespace openpni::io
