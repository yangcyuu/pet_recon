#include "include/experimental/node/Senmaps.hpp"

#include <algorithm>
#include <iostream>
#include <list>
#include <mutex>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

#include "impl/Copy.h"
#include "impl/MichCrystal.h"
#include "impl/Projection.h"
#include "impl/Share.hpp"
#include "impl/WrappedConv3D.h"
#include "include/experimental/node/MichNorm.hpp"

using Grids3f = openpni::experimental::core::Grids<3, float>;
#ifndef MichInfoHub
#define MichInfoHub(m) core::MichInfoHub::create(m)
#endif
#ifndef IndexConverter
#define IndexConverter(m) core::IndexConverter::create(m)
#endif
#ifndef RangeGenerator
#define RangeGenerator(m) core::RangeGenerator::create(m)
#endif
namespace openpni::experimental::node::impl {

void h_fix_senmap_value(
    std::span<float> __senmap_values) {
  const auto maxValue = *std::max_element(__senmap_values.begin(), __senmap_values.end());
  for (auto &value : __senmap_values)
    value = senmap_fix_value(value, maxValue);
}

std::unique_ptr<float[]> h_createSenmap_impl(
    core::Grids<3, float> const &__grids, MichCrystal &__michCrystal, interface::Conv3D &__convolver,
    MichNormalization *__normalization, MichAttn *__attn, LORBatch &lorBatch) {
  auto result = std::make_unique_for_overwrite<float[]>(__grids.totalSize());
  std::fill_n(result.get(), __grids.totalSize(), 0.0f);
  const auto michDefine = __michCrystal.mich();

  std::vector<core::MichStandardEvent> tempMichEvents;
  for (auto lors = lorBatch.nextHBatch(); !lors.empty(); lors = lorBatch.nextHBatch()) {
    if (lors.size() > tempMichEvents.size())
      tempMichEvents.resize(lors.size());
    impl::h_fill_crystal_ids(tempMichEvents.data(), lors.data(), lors.size(), michDefine);
    if (__normalization) {
      auto normFactors = __normalization->getHNormFactorsBatch(
          std::span<core::MichStandardEvent const>(tempMichEvents.data(), lors.size()));
      tools::parallel_for_each(lors.size(),
                               [&](std::size_t index) { tempMichEvents[index].value *= normFactors[index]; });
    }
    if (__attn) {
      auto attnFactors =
          __attn->getHAttnFactorsBatch(std::span<core::MichStandardEvent const>(tempMichEvents.data(), lors.size()));
      tools::parallel_for_each(lors.size(),
                               [&](std::size_t index) { tempMichEvents[index].value *= attnFactors[index]; });
    }
    __michCrystal.fillHCrystalsBatch(tempMichEvents);
    tools::parallel_for_each(lors.size(), [&](std::size_t index) {
      float bias = static_cast<float>(core::instant_random_float(index));
      openpni::experimental::node::impl::simple_reverse_path_integral(
          bias, 1.5, core::TensorDataOutput<float, 3>{__grids, result.get()}, tempMichEvents[index]);
    });
  }
  __convolver.convH(core::TensorDataIO<float, 3>{__grids, result.get(), result.get()});
  h_fix_senmap_value(std::span<float>(result.get(), __grids.totalSize()));
  return result;
}
cuda_sync_ptr<float> d_createSenmap_impl(
    core::Grids<3, float> __grids, MichCrystal &__michCrystal, interface::Conv3D &__convolver,
    MichNormalization *__normalization, MichAttn *__attn, LORBatch &lorBatch) {
  auto result = make_cuda_sync_ptr<float>(__grids.totalSize(), "Senmap_result");
  result.allocator().memset(0, result.span());
  const auto michDefine = __michCrystal.mich();

  cuda_sync_ptr<core::MichStandardEvent> tempMichEvents{"Senmap_tempMichEvents"};
  for (auto lors = lorBatch.nextDBatch(); !lors.empty(); lors = lorBatch.nextDBatch()) {
    tempMichEvents.reserve(lors.size());
    impl::d_fill_standard_events_ids_from_lor_ids(tempMichEvents.data(), lors, michDefine);
    __michCrystal.fillDCrystalsBatch(tempMichEvents);
    if (__normalization)
      impl::d_mul_standard_events_values(
          tempMichEvents.get(), __normalization->getDNormFactorsBatch(tempMichEvents.cspan(lors.size())), lors.size());
    if (__attn)
      impl::d_mul_standard_events_values(tempMichEvents.get(),
                                         __attn->getDAttnFactorsBatch(tempMichEvents.cspan(lors.size())), lors.size());
    impl::d_simple_path_reverse_integral_batch(core::TensorDataOutput<float, 3>{__grids, result.get()},
                                               tempMichEvents.cspan(lors.size()), 2.5);
  }
  __convolver.convD(core::TensorDataIO<float, 3>{__grids, result.get(), result.get()});
  d_fix_senmap_value(std::span<float>(result.get(), __grids.totalSize()));

  return result;
}
} // namespace openpni::experimental::node::impl

namespace openpni::experimental::node {
class MichSenmap_impl {
  struct SenmapItem {
    SenmapItem(
        Grids3f __grids, int __subsetId, MichSenmap::SenmapMode __mode)
        : grids(__grids)
        , subsetId(__subsetId)
        , mode(__mode) {}
    SenmapItem(
        const SenmapItem &other)
        : grids(other.grids)
        , subsetId(other.subsetId)
        , mode(other.mode) {
      h_data = other.h_data; // If other has h_data, copy pointer directly.
      if (!h_data)           // If other has no h_data, then try to copy other.d_data to this.h_data.
        if (other.d_data)    // In normal condition, other.d_data and other.h_data will not exist at the same time.
          h_data = device_deep_copy_to_host(other.d_data);
        else // It is abnormal that both other.d_data and other.h_data do not exist.
          throw exceptions::algorithm_unexpected_condition(
              "Internal Error: MichSenmap::SenmapItem copy constructor: no data");
    }
    SenmapItem(SenmapItem &&other) = default;

    SenmapItem &operator=(
        const SenmapItem &other) {
      if (this != &other) {
        grids = other.grids;
        subsetId = other.subsetId;
        mode = other.mode;
        h_data = other.h_data; // If other has h_data, copy pointer directly.
        if (!h_data)           // If other has no h_data, then try to copy other.d_data to this.h_data.
          if (other.d_data)    // In normal condition, other.d_data and other.h_data will not exist at the same time.
            h_data = device_deep_copy_to_host(other.d_data);
          else // It is abnormal that both other.d_data and other.h_data do not exist.
            throw exceptions::algorithm_unexpected_condition(
                "Internal Error: MichSenmap::SenmapItem copy assignment operator: no data");
      }
      return *this;
    }
    SenmapItem &operator=(SenmapItem &&other) = default;
    bool equals(
        const Grids3f &otherGrid, int otherId, MichSenmap::SenmapMode otherMode) const {
      if (mode == otherMode && mode == MichSenmap::Mode_listmode)
        return grids == otherGrid;
      else
        return subsetId == otherId && grids == otherGrid;
    }
    bool equals(
        const SenmapItem &other) const {
      return equals(other.grids, other.subsetId, other.mode);
    }
    auto getGrids() const { return grids; }

  private:
    Grids3f grids;
    int subsetId;
    MichSenmap::SenmapMode mode;

  public:
    std::shared_ptr<float[]> h_data;
    cuda_sync_ptr<float> d_data;
  };
  using SenmapBufferItem = std::list<SenmapItem>::iterator;

public:
  MichSenmap_impl(
      const core::MichDefine &__mich, interface::Conv3D &__conv3D)
      : m_conv3D(__conv3D)
      , m_lorBatch(__mich)
      , m_michCrystal(__mich) {
    m_lorBatch.setBinCut(0);
    m_lorBatch.setMaxRingDiff(-1);
    m_lorBatch.setSubsetNum(1);
  }
  ~MichSenmap_impl() = default;

  std::unique_ptr<MichSenmap_impl> copy(
      interface::Conv3D &__newConv3D) {
    std::lock_guard lock(m_mutex);
    auto result = std::make_unique<MichSenmap_impl>(m_michCrystal.mich(), __newConv3D);
    result->m_subsetNum = m_subsetNum;
    result->m_maxBufferedImages = m_maxBufferedImages;
    for (auto iter = m_senmaps.begin(); iter != m_senmaps.end(); ++iter)
      if (!iter->h_data)
        borrow_senmap_from_device(iter); // Make sure all senmaps are host data.
    result->m_senmaps = m_senmaps;       // When copying, all senmaps are host data.
    result->m_normalization = nullptr;   // Do not copy normalization binding.
    result->m_attn = nullptr;            // Do not copy attenuation binding.
    result->m_source = m_source;
    result->m_mode = m_mode;
    return result;
  }
  void join(
      MichSenmap_impl *other) {
    std::lock_guard ___(this->m_mutex);
    std::lock_guard ____(other->m_mutex);
    for (auto iterOther = other->m_senmaps.begin(); iterOther != other->m_senmaps.end(); ++iterOther) {
      auto iterThis = std::find_if(m_senmaps.begin(), m_senmaps.end(),
                                   [&](const SenmapItem &item) { return item.equals(*iterOther); });
      if (iterThis == m_senmaps.end()) {
        if (m_senmaps.size() >= static_cast<std::size_t>(m_maxBufferedImages))
          m_senmaps.pop_front();
        m_senmaps.push_back(*iterOther);
      }
    }
  }

  void setSubsetNum(
      int num) {
    m_subsetNum = std::max(1, num);
    m_maxBufferedImages = std::max(m_subsetNum, m_maxBufferedImages);
    m_senmaps.clear(); // clear: means not calculated, should be recalculated later.
    if (m_mode == MichSenmap::Mode_mich)
      m_lorBatch.setSubsetNum(m_subsetNum);
    else
      m_lorBatch.setSubsetNum(1);
  }
  void setMaxBufferedImages(
      int num) {
    m_maxBufferedImages = std::max(num, m_subsetNum);
    while (m_senmaps.size() > static_cast<std::size_t>(m_maxBufferedImages))
      m_senmaps.pop_back();
  }
  void setPreferredSource(
      MichSenmap::SenmapSource source) {
    m_source = source;
  }
  void setMode(
      MichSenmap::SenmapMode mode) {
    m_mode = mode;
    m_senmaps.clear(); // clear: means not calculated, should be recalculated later.
  }

  std::unique_ptr<float[]> dumpHSenmap(
      int subsetIndex, Grids3f grids) {
    if (subsetIndex < 0 || subsetIndex >= m_subsetNum)
      throw exceptions::algorithm_unexpected_condition("MichSenmap::getSenmap: subsetIndex out of range");
    return this->h_deepCopy(find_senmap_ifnot_create(grids, subsetIndex));
  }

  cuda_sync_ptr<float> dumpDSenmap(
      int subsetIndex, Grids3f grids) {
    if (subsetIndex < 0 || subsetIndex >= m_subsetNum)
      throw exceptions::algorithm_unexpected_condition("MichSenmap::getSenmap: subsetIndex out of range");
    return this->d_deepCopy(find_senmap_ifnot_create(grids, subsetIndex));
  }

#define UPDATE_MEASUREMENT 0
  void updateHImage(
      float *h_updateImage, float *h_out, int subsetIndex, core::Grids<3, float> grids) {
    auto iter = find_senmap_ifnot_create(grids, subsetIndex);
    if (!iter->h_data)
      borrow_senmap_from_device(iter);
    tools::parallel_for_each(grids.totalSize(),
                             [&](std::size_t index) { h_out[index] *= h_updateImage[index] / iter->h_data[index]; });
    auto max_value = *std::max_element(h_out, h_out + grids.totalSize());
    auto min_value = max_value * 1e-7;
    tools::parallel_for_each(
        grids.totalSize(), [&](std::size_t index) { h_out[index] = core::FMath<float>::max(h_out[index], min_value); });
#if UPDATE_MEASUREMENT
    m_lastUpdateMeasurement = impl::h_cal_update_measurements(h_updateImage, iter->h_data.get(), grids.totalSize());
#endif
  }
  void updateDImage(
      float *d_updateImage, float *d_out, int subsetIndex, core::Grids<3, float> grids) {
    auto iter = find_senmap_ifnot_create(grids, subsetIndex);
    if (!iter->d_data)
      borrow_senmap_from_host(iter);
    impl::d_image_update(d_updateImage, d_out, iter->d_data.get(), grids.totalSize());
#if UPDATE_MEASUREMENT
    m_lastUpdateMeasurement = impl::d_cal_update_measurements(d_updateImage, iter->d_data.get(), grids.totalSize());
#endif
  }
#undef UPDATE_MEASUREMENT

  void bindNormalization(
      MichNormalization *normalization) {
    if (m_normalization == normalization)
      return;
    m_normalization = normalization;
    while (m_senmaps.size() > 0)
      m_senmaps.pop_back(); // clear: means not calculated, should be recalculated later.
  }
  void bindAttenuation(
      MichAttn *attn) {
    if (m_attn == attn)
      return;
    m_attn = attn;
    while (m_senmaps.size() > 0)
      m_senmaps.pop_back(); // clear: means not calculated, should be recalculated later.
  }
  void preBaking(
      std::vector<std::pair<Grids3f, int>> const &gridsAndSubsetList) {
    m_maxBufferedImages = std::max(m_maxBufferedImages, static_cast<int>(gridsAndSubsetList.size()));
    for (const auto &[grids, subsetIndex] : gridsAndSubsetList)
      find_senmap_ifnot_create(grids, subsetIndex);
  }
  void clearCache() { m_senmaps.clear(); }
  float last_update_measurement() const { return m_lastUpdateMeasurement; }

private:
  std::unique_ptr<float[]> h_createSenmap(
      Grids3f grids, int subsetIndex) {
    PNI_DEBUG(std::format("Creating senmap cpu for subset {}, size {}x{}x{}\n", subsetIndex, grids.size.dimSize[0],
                          grids.size.dimSize[1], grids.size.dimSize[2]));
    if (m_mode == MichSenmap::Mode_mich) {
      m_lorBatch.setCurrentSubset(subsetIndex);
      return impl::h_createSenmap_impl(grids, m_michCrystal, m_conv3D, m_normalization, m_attn, m_lorBatch);
    } else {
      m_lorBatch.setSubsetNum(1);
      m_lorBatch.setCurrentSubset(0); // In listmode mode, always use subset 0.
      auto result = impl::h_createSenmap_impl(grids, m_michCrystal, m_conv3D, m_normalization, m_attn, m_lorBatch);
      tools::parallel_for_each(grids.totalSize(),
                               [&](std::size_t index) { result[index] /= static_cast<float>(m_subsetNum); });
      return result;
    }
  }
  cuda_sync_ptr<float> d_createSenmap(
      Grids3f grids, int subsetIndex) {
    PNI_DEBUG(std::format("Creating senmap gpu for subset {}, size {}x{}x{}\n", subsetIndex, grids.size.dimSize[0],
                          grids.size.dimSize[1], grids.size.dimSize[2]));
    if (m_mode == MichSenmap::Mode_mich) {
      m_lorBatch.setSubsetNum(m_subsetNum);
      m_lorBatch.setCurrentSubset(subsetIndex);
      return impl::d_createSenmap_impl(grids, m_michCrystal, m_conv3D, m_normalization, m_attn, m_lorBatch);
    } else {
      m_lorBatch.setSubsetNum(1);
      m_lorBatch.setCurrentSubset(0); // In listmode mode, always use
      auto result = impl::d_createSenmap_impl(grids, m_michCrystal, m_conv3D, m_normalization, m_attn, m_lorBatch);
      impl::d_vector_divide(result.span(), m_subsetNum);
      return result;
    }
  }
  SenmapBufferItem find_senmap_item(
      const Grids3f &__grids, int subsetIndex) {
    return std::find_if(m_senmaps.begin(), m_senmaps.end(),
                        [&](const SenmapItem &item) { return item.equals(__grids, subsetIndex, m_mode); });
  }
  SenmapBufferItem insert_senmap_item(
      const Grids3f &__grids, int subsetIndex) {
    if (m_senmaps.size() >= static_cast<std::size_t>(m_maxBufferedImages))
      m_senmaps.pop_back();
    return m_senmaps.insert(m_senmaps.begin(), createSenmap(__grids, subsetIndex));
  }
  SenmapBufferItem find_senmap_ifnot_create(
      const Grids3f &__grids, int subsetIndex) {
    if (auto iter = find_senmap_item(__grids, subsetIndex); iter != m_senmaps.end())
      return iter;
    return insert_senmap_item(__grids, subsetIndex);
  }
  SenmapItem createSenmap(
      Grids3f grids, int subsetIndex) {
    SenmapItem item{grids, subsetIndex, m_mode};
    if (m_source == MichSenmap::Senmap_GPU) {
      item.d_data = d_createSenmap(grids, subsetIndex);
    } else {
      item.h_data = h_createSenmap(grids, subsetIndex);
    }
    return item;
  }
  std::unique_ptr<float[]> h_deepCopy(
      SenmapBufferItem iter) {
    if (iter->h_data) // Check CPU data first
      return host_deep_copy(iter->h_data.get(), iter->getGrids().totalSize());
    else if (iter->d_data) // If no CPU data, check GPU data
      return device_deep_copy_to_host(iter->d_data);
    else
      throw exceptions::algorithm_unexpected_condition("Internal Error: MichSenmap::h_deepCopy: no data");
    return {};
  }
  cuda_sync_ptr<float> d_deepCopy(
      SenmapBufferItem iter) {
    if (iter->d_data)
      return device_deep_copy(iter->d_data);
    else if (iter->h_data)
      return openpni::make_cuda_sync_ptr_from_hcopy(
          std::span<float const>(iter->h_data.get(), iter->getGrids().totalSize()), "Senmap_copy_to_device");
    else
      throw exceptions::algorithm_unexpected_condition("Internal Error: MichSenmap::d_deepCopy: no data");
    return {};
  }
  void borrow_senmap_from_host(
      SenmapBufferItem iter) {
    if (!iter->h_data)
      throw exceptions::algorithm_unexpected_condition(
          "Internal Error: MichSenmap::borrow_senmap_from_host: no host data");
    iter->d_data = openpni::make_cuda_sync_ptr_from_hcopy(
        std::span<float const>(iter->h_data.get(), iter->getGrids().totalSize()), "Senmap_borrow_to_device");
  }
  void borrow_senmap_from_device(
      SenmapBufferItem iter) {
    if (!iter->d_data)
      throw exceptions::algorithm_unexpected_condition(
          "Internal Error: MichSenmap::borrow_senmap_from_device: no device data");
    iter->h_data = device_deep_copy_to_host(iter->d_data);
  }

private:
  interface::Conv3D &m_conv3D;
  LORBatch m_lorBatch;
  MichCrystal m_michCrystal;
  int m_subsetNum = 1;
  int m_maxBufferedImages = 1;
  std::list<SenmapItem> m_senmaps;
  MichNormalization *m_normalization{nullptr};
  MichAttn *m_attn{nullptr};
  MichSenmap::SenmapSource m_source = MichSenmap::Senmap_GPU;
  MichSenmap::SenmapMode m_mode = MichSenmap::Mode_mich;
  std::recursive_mutex m_mutex;
  float m_lastUpdateMeasurement;
};

MichSenmap::MichSenmap(
    interface::Conv3D &__conv3D, const core::MichDefine &__mich)
    : m_impl(std::make_unique<MichSenmap_impl>(__mich, __conv3D)) {}

MichSenmap::~MichSenmap() = default;

void MichSenmap::setSubsetNum(
    int num) {
  m_impl->setSubsetNum(num);
}

std::unique_ptr<float[]> MichSenmap::dumpHSenmap(
    int subsetIndex, core::Grids<3, float> grids) {
  return m_impl->dumpHSenmap(subsetIndex, grids);
}

void MichSenmap::bindNormalization(
    MichNormalization *normalization) {
  m_impl->bindNormalization(normalization);
}

void MichSenmap::updateHImage(
    float *h_updateImage, float *h_out, int subsetIndex, core::Grids<3, float> grids) {
  m_impl->updateHImage(h_updateImage, h_out, subsetIndex, grids);
}

void MichSenmap::updateDImage(
    float *d_updateImage, float *d_out, int subsetIndex, core::Grids<3, float> grids) {
  m_impl->updateDImage(d_updateImage, d_out, subsetIndex, grids);
}

void MichSenmap::setMaxBufferedImages(
    int num) {
  m_impl->setMaxBufferedImages(num);
}

void MichSenmap::setPreferredSource(
    SenmapSource source) {
  m_impl->setPreferredSource(source);
}
void MichSenmap::setMode(
    SenmapMode mode) {
  m_impl->setMode(mode);
}
void MichSenmap::preBaking(
    std::vector<std::pair<Grids3f, int>> const &gridsAndSubsetList) {
  m_impl->preBaking(gridsAndSubsetList);
}
void MichSenmap::clearCache() {
  m_impl->clearCache();
}

MichSenmap MichSenmap::copy(
    interface::Conv3D &__conv3D) {
  return MichSenmap(m_impl->copy(__conv3D));
}
void MichSenmap::join(
    MichSenmap *other) {
  m_impl->join(other->m_impl.get());
}
float MichSenmap::lastUpdateMeasurement() {
  return m_impl->last_update_measurement();
}
void MichSenmap::bindAttenuation(
    MichAttn *attn) {
  m_impl->bindAttenuation(attn);
}

MichSenmap::MichSenmap(
    std::unique_ptr<MichSenmap_impl> &&impl)
    : m_impl(std::move(impl)) {}
} // namespace openpni::experimental::node