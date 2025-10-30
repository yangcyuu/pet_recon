#pragma once
#include "../PnI-Config.hpp"
#include "../math/Geometry.hpp"
#include "PetDataType.h"

namespace openpni {
namespace process {
// LOR索引重排器如何定义，请看下面的concept
template <typename T>
concept ReArrangerContract = requires(const T &rearranger, std::size_t lorIndex, uint32_t _1, uint32_t _2) {
  {
    rearranger(lorIndex)
  } -> std::same_as<std::pair<std::uint32_t,
                              std::uint32_t>>;                // 需要重载一个函数将自定义LOR索引转换为两个晶体的索引
  { rearranger(_1, _2) } -> std::same_as<std::size_t>;        // 需要重载一个函数将两个晶体的索引转换为自定义LOR索引
  { rearranger.michSize() } -> std::same_as<std::size_t>;     // 需要重载一个函数返回自定义LOR索引的总数
  { rearranger.crystalNum() } -> std::same_as<std::uint32_t>; // 需要重载一个函数返回晶体的总数
};

// LOR索引重排器如何定义，请看下面的concept
template <typename T>
concept IndexerContract = requires(const T &rearranger, std::size_t dataIndex, uint32_t _1, uint32_t _2) {
  { rearranger(dataIndex) } -> std::same_as<basic::Vec2<std::uint32_t>>;
  { rearranger[dataIndex] } -> std::same_as<std::size_t>;
};

} // namespace process
} // namespace openpni

namespace openpni::basic {
struct CrystalInfo {
  const CrystalGeometry *geometry; // 晶体位置，单位毫米
  const int16_t *tof_deviation;    // TOF信息的标准差，单位皮秒，nullptr表示没有TOF信息(= inf)
  const int16_t *tof_mean;         // TOF信息的平均值，单位皮秒，nullptr表示没有TOF信息(= 0)
  uint32_t crystalIndex;           // 晶体的索引(可选)
};

template <typename _GainType>
struct Event {
  using GainType = _GainType;

  CrystalInfo crystal1;   // 第一个晶体的信息
  CrystalInfo crystal2;   // 第二个晶体的信息
  const int16_t *time1_2; // 时间信息，到达时间1减去到达时间2，单位皮秒，nullptr表示没有时间信息
  const GainType *gain;   // 增益信息，mich数据等于mich值，listmode数据等于1
  size_t eventIndex;      // 事件在数据集中的索引(可能用于伪随机数)
};

template <typename T> // Dataset是管理数据的视图（方便访问数据，但是不持有数据的生命周期）
concept DataView = requires(const T &dataset, std::size_t index) {
  { dataset.size() } -> std::same_as<std::size_t>;                    // 数据集的大小
  { dataset.at(index) } -> std::same_as<Event<typename T::GainType>>; // 输入数据编号，返回数据内容
};

template <process::IndexerContract _Indexer, typename _QTYValueType>
struct DataViewQTY // Data View Quantity statistic
{
  using GainType = _QTYValueType;
  using _Event = Event<_QTYValueType>;

  const GainType *qtyValue;               // 事件计数量
  const CrystalGeometry *crystalGeometry; // 晶体位置，单位毫米
  _Indexer indexer;

  __PNI_CUDA_MACRO__ std::size_t size() const { return indexer.count(); }
  __PNI_CUDA_MACRO__ std::size_t crystalNum() const {
    return indexer.crystalNum(); // 返回晶体的总数
  }
  __PNI_CUDA_MACRO__ CrystalInfo crystal(
      std::size_t __index) const {
    CrystalInfo result;
    result.geometry = crystalGeometry + __index;
    result.tof_deviation = nullptr; // QTY数据集不包含TOF信息
    result.tof_mean = nullptr;      // QTY数据集不包含TOF信息
    result.crystalIndex = __index;
    return result;
  }
  __PNI_CUDA_MACRO__ _Event at(
      std::size_t index) const // 从 子集的LOR 到 event
  {
    _Event result;
    const auto [crystal1, crystal2] = indexer(index);
    result.crystal1 = crystal(crystal1);
    result.crystal2 = crystal(crystal2);
    result.time1_2 = nullptr; // QTY数据集不包含时间信息
    result.gain = qtyValue ? qtyValue + indexer[index] : nullptr;
    result.eventIndex = index;
    return result;
  }
};

struct DataViewListmodePlain // Data View Quantity statistic(listmode without TOF)
{
  using GainType = float;
  using _Event = Event<float>;

  const CoinListmode *listmodes;
  std::size_t count;
  std::size_t crystals;
  const CrystalGeometry *crystalGeometry;

  __PNI_CUDA_MACRO__ std::size_t size() const { return count; }
  __PNI_CUDA_MACRO__ std::size_t crystalNum() const { return crystals; }
  __PNI_CUDA_MACRO__ CrystalInfo crystal(
      std::size_t __index) const {
    CrystalInfo result;
    result.geometry = crystalGeometry + __index;
    result.tof_deviation = nullptr; // PlainListmode数据集不包含TOF信息
    result.tof_mean = nullptr;      // PlainListmode数据集不包含TOF信息
    result.crystalIndex = __index;
    return result;
  }
  __PNI_CUDA_MACRO__ _Event at(
      std::size_t index) const // 从 子集的LOR 到 event
  {
    _Event result;
    result.crystal1 = crystal(listmodes[index].globalCrystalIndex1);
    result.crystal2 = crystal(listmodes[index].globalCrystalIndex2);
    result.time1_2 = &listmodes[index].time1_2pico;
    result.gain = nullptr;
    result.eventIndex = index;
    return result;
  }
};

template <FloatingPoint_c _PointPrecision = float>
struct DataViewListmodeTOF {
  using GainType = float;
  using _Event = Event<float>;

  const CoinListmode *listmodes;
  std::size_t count;
  std::size_t crystals;
  const CrystalGeometry *crystalGeometry;
  const int16_t *crystalTOFDeviation; // TOF信息的标准差，单位皮秒，nullptr表示没有TOF信息(= inf)
  const int16_t *crystalTOFMean;      // TOF信息的平均值，单位皮秒，nullptr表示没有TOF信息(= 0)

  __PNI_CUDA_MACRO__ std::size_t size() const { return count; }
  __PNI_CUDA_MACRO__ std::size_t crystalNum() const { return crystals; }
  __PNI_CUDA_MACRO__ CrystalInfo crystal(
      std::size_t __index) const {
    CrystalInfo result;
    result.geometry = crystalGeometry + __index;
    result.tof_deviation =
        crystalTOFDeviation ? crystalTOFDeviation + __index : nullptr;     // ListmodeTof数据集可能含TOF信息
    result.tof_mean = crystalTOFMean ? crystalTOFMean + __index : nullptr; // ListmodeTof数据集可能含TOF信息
    result.crystalIndex = __index;
    return result;
  }
  __PNI_CUDA_MACRO__ _Event at(
      std::size_t index) const // 从 子集的LOR 到 event
  {
    _Event result;
    result.crystal1 = crystal(listmodes[index].globalCrystalIndex1);
    result.crystal2 = crystal(listmodes[index].globalCrystalIndex2);
    result.time1_2 = &listmodes[index].time1_2pico;
    result.gain = nullptr;
    result.eventIndex = index;
    return result;
  }
};

enum class FactorType : int { Multiply, Addition };
template <FactorType _FactorType, process::IndexerContract _Indexer, typename _FactorValue = float>
struct _FactorAdaptorMich {

  _FactorValue *factor; // use __Indexer to get index
  _Indexer indexer;

  __PNI_CUDA_MACRO__ _FactorValue operator()(
      const Event<_FactorValue> event) const {
    if constexpr (_FactorType == FactorType::Multiply)
      return factor ? *(factor + indexer.indexInMich(event.crystal1.crystalIndex, event.crystal2.crystalIndex)) : 1;
    else if constexpr (_FactorType == FactorType::Addition)
      return factor ? *(factor + indexer.indexInMich(event.crystal1.crystalIndex, event.crystal2.crystalIndex)) : 0;
    else {
      static_assert([]() { return false; }(), "Unknown FactorType");
      return _FactorValue(0);
    }
  }
};

} // namespace openpni::basic
