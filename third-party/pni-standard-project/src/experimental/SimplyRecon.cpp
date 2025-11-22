#include "include/experimental/example/SimplyRecon.hpp"

#include <fstream>

#include "impl/Share.hpp"
#include "impl/WrappedConv3D.h"
#include "include/basic/Matrix.hpp"
#include "include/detector/BDM2.hpp"
#include "include/experimental/algorithms/CalGeometry.hpp"
#include "include/experimental/algorithms/PathIntegral.hpp"
#include "include/experimental/example/EasyParallel.hpp"
#include "include/experimental/node/GaussConv3D.hpp"
#include "include/experimental/node/KGConv3D.hpp"
#include "include/experimental/node/KNNConv3D.hpp"
#include "include/experimental/node/LORBatch.hpp"
#include "include/experimental/node/MichCrystal.hpp"
#include "include/experimental/node/Senmaps.hpp"
#include "include/experimental/tools/Loop.hpp"
#include "include/io/Decoding.hpp"
#include "include/io/IO.hpp"
#include "include/misc/CycledBuffer.hpp"
#include "include/misc/ListmodeBuffer.hpp"
#include "include/process/FBP.hpp"
#include "include/process/FDK.hpp"
#include "src/common/Debug.h"
#include "src/experimental/impl/Projection.h"

#ifndef MichInfoHub
#define MichInfoHub(m) core::MichInfoHub::create(m)
#endif
#ifndef IndexConverter
#define IndexConverter(m) core::IndexConverter::create(m)
#endif
#ifndef RangeGenerator
#define RangeGenerator(m) core::RangeGenerator::create(m)
#endif
namespace openpni::experimental::example {
std::size_t _GB(
    unsigned long long value) {
  return value * 1024 * 1024 * 1024;
}
std::size_t find_value(
    std::size_t __total, std::size_t __maxChunk, std::size_t __supposedDivisor) {
  if (__total <= __maxChunk)
    return __total;
  if (__total <= __maxChunk * __supposedDivisor)
    return __total / __supposedDivisor + (__total % __supposedDivisor != 0 ? 1 : 0);

  auto chunks = __total / __maxChunk + (__total % __maxChunk != 0 ? 1 : 0);
  auto chunkNum = chunks % __supposedDivisor == 0 ? chunks : chunks + (__supposedDivisor - chunks % __supposedDivisor);
  return __total / chunkNum + (__total % chunkNum != 0 ? 1 : 0);
}

inline auto generatePairs(
    std::size_t __max, std::size_t __count) -> std::vector<std::pair<std::size_t, std::size_t>> {
  std::vector<std::size_t> indices;
  indices.reserve(__count + 1); // 性能优化
  std::vector<std::pair<std::size_t, std::size_t>> result;
  result.reserve(__count); // 性能优化

  for (std::size_t i = 0; i < __count; i++)
    indices.push_back(i * __max / __count);
  indices.push_back(__max);

  for (std::size_t i = 0; i < __count; i++)
    result.push_back(std::make_pair(indices[i], indices[i + 1]));

  return result;
};

void fix_integral_value(
    std::span<float> values) {
  const auto maxValue = *std::max_element(values.begin(), values.end());
  for (auto &value : values)
    if (value < maxValue * 1e-7f)
      value = 0;
    else
      value = 1.f / value;
}

void instant_OSEM_mich_CPU(
    core::Image3DOutput<float> out_OSEMImg, OSEM_params params, interface::Conv3D &conv3D, float const *michValue,
    node::MichNormalization *michNorm, node::MichRandom *michRand, node::MichScatter *michScat,
    core::MichDefine const &mich) {
  node::MichSenmap senmap(conv3D, mich);
  senmap.setSubsetNum(params.subsetNum);
  senmap.bindNormalization(michNorm);
  senmap.setPreferredSource(node::MichSenmap::Senmap_CPU);
  node::LORBatch lorBatch(mich);
  lorBatch.setSubsetNum(params.subsetNum).setBinCut(params.binCutRatio * MichInfoHub(mich).getBinNum());
  node::MichCrystal michCrystal(mich);

  std::fill_n(out_OSEMImg.ptr, out_OSEMImg.grid.totalSize(), 1.0f);
  std::vector<float> tempValues;

  std::unique_ptr temp1 = std::make_unique_for_overwrite<float[]>(out_OSEMImg.grid.totalSize());
  std::unique_ptr temp2 = std::make_unique_for_overwrite<float[]>(out_OSEMImg.grid.totalSize());
  std::vector<core::MichStandardEvent> temp_event;

  for (const auto iteration : std::views::iota(0, params.iterNum))
    for (const auto subsetId : std::views::iota(0, params.subsetNum)) {
      conv3D.convH(core::Image3DIO<float>{out_OSEMImg.grid, out_OSEMImg.ptr, temp1.get()});
      std::fill_n(temp2.get(), out_OSEMImg.grid.totalSize(), 0.0f);
      for (auto lors = lorBatch.setCurrentSubset(subsetId).nextHBatch(); !lors.empty(); lors = lorBatch.nextHBatch()) {
        if (lors.size() > temp_event.size())
          temp_event.resize(lors.size());
        tools::parallel_for_each(lors.size(), [&](std::size_t index) {
          auto [cry1, cry2] = IndexConverter(mich).getCrystalIDFromLORID(lors[index]);
          temp_event[index].crystal1 = cry1;
          temp_event[index].crystal2 = cry2;
        });
        michCrystal.fillHCrystalsBatch(std::span<core::MichStandardEvent>{temp_event.data(), lors.size()});
        tempValues.resize(lors.size());
        tools::parallel_for_each(lors.size(), [&](std::size_t index) {
          tempValues[index] = node::impl::simple_path_integral(core::instant_random_float(index), params.sample_rate,
                                                               core::Image3DInput<float>{out_OSEMImg.grid, temp1.get()},
                                                               temp_event[index].geo1.O, temp_event[index].geo2.O);
        });
        if (michRand || michScat)
          node::impl::h_apply_correction_factor(tempValues, temp_event, michNorm, michRand, michScat);
        fix_integral_value(tempValues);
        tools::parallel_for_each(lors.size(), [&](std::size_t index) { tempValues[index] *= michValue[lors[index]]; });
        tools::parallel_for_each(lors.size(), [&](std::size_t index) {
          node::impl::simple_reverse_path_integral(core::instant_random_float(index), params.sample_rate,
                                                   tempValues[index],
                                                   core::Image3DOutput<float>{out_OSEMImg.grid, temp2.get()},
                                                   temp_event[index].geo1.O, temp_event[index].geo2.O);
        });
      }

      conv3D.deconvH(core::Image3DIO<float>{out_OSEMImg.grid, temp2.get(), temp2.get()});
      senmap.updateHImage(temp2.get(), out_OSEMImg.ptr, subsetId, out_OSEMImg.grid);

      PNI_DEBUG(std::format("OSEM iteration {}, subset {}\n", iteration, subsetId));
      break;
    }
}
auto average_variant(
    auto Range) {
  if (Range.empty())
    return 0.0;
  auto sum = std::accumulate(Range.begin(), Range.end(), 0.0, [](auto a, auto b) { return a + b; });
  auto average = sum / static_cast<double>(std::distance(Range.begin(), Range.end()));
  auto variance = std::accumulate(Range.begin(), Range.end(), 0.0, [average](auto a, auto b) {
    return a + (static_cast<double>(b) - average) * (static_cast<double>(b) - average);
  });
  variance /= static_cast<double>(std::distance(Range.begin(), Range.end()));
  return std::sqrt(variance) / average;
}
struct OSEM_context {
  std::unique_ptr<node::LORBatch> lorBatch;
  std::unique_ptr<node::MichCrystal> michCrystal;
  std::unique_ptr<node::MichSenmap> senmap;
  cuda_sync_ptr<float> d_michValue;
};

struct OSEM_context_listmode {
  std::size_t events_count;
  cuda_sync_ptr<openpni::basic::Listmode_t> Listmode_data;
  std::size_t batchSize = 1024 * 1024 * 5; // 5M
  std::unique_ptr<node::MichCrystal> michCrystal;
  std::unique_ptr<node::MichSenmap> senmap;
  interface::Conv3D *convolver;
};

void osem_impl_CUDA(
    core::Image3DOutput<float> d_out_OSEMImg, OSEM_params params, interface::Conv3D &conv3D,
    node::MichNormalization *michNorm, node::MichRandom *michRand, node::MichScatter *michScat,
    node::MichAttn *michAttn, core::MichDefine const &mich, OSEM_context &context) {
  openpni::cuda_sync_ptr<float> d_tempValues{"OSEM_tempValues"};
  cuda_sync_ptr<float> d_tempMichValue{"OSEM_tempMichValue"};
  cuda_sync_ptr<float> d_tmp1 = openpni::make_cuda_sync_ptr<float>(d_out_OSEMImg.grid.totalSize(), "OSEM_tmp1");
  cuda_sync_ptr<float> d_tmp2 = openpni::make_cuda_sync_ptr<float>(d_out_OSEMImg.grid.totalSize(), "OSEM_tmp2");
  cuda_sync_ptr<core::MichStandardEvent> d_tempEvents{"OSEM_tempEvents"};

  cuda_sync_ptr<float> d_factors_mul{"OSEM_MulFactor"};
  cuda_sync_ptr<float> d_factors_add{"OSEM_AddFactor"};

  auto &lorBatch = *context.lorBatch;
  auto &michCrystal = *context.michCrystal;
  auto &senmap = *context.senmap;
  auto &d_michValue = context.d_michValue;

  for (const auto iteration : std::views::iota(0, params.iterNum))
    for (const auto subsetId : std::views::iota(0, params.subsetNum)) {
      PNI_DEBUG(std::format("OSEM iteration {}, subset {}\n", iteration, subsetId));

      conv3D.convD(core::Image3DIO<float>{d_out_OSEMImg.grid, d_out_OSEMImg.ptr, d_tmp1.get()});
      d_tmp2.memset(0);
      for (auto lors = lorBatch.setCurrentSubset(subsetId).nextDBatch(); !lors.empty(); lors = lorBatch.nextDBatch()) {
        d_tempValues.reserve(lors.size());
        d_tempMichValue.reserve(lors.size());
        d_tempEvents.reserve(lors.size());
        node::impl::d_redirect_from_mich(lors.size(), lors.data(), d_michValue, d_tempMichValue.get());
        node::impl::d_fill_standard_events_ids_from_lor_ids(d_tempEvents, lors, mich);
        michCrystal.fillDCrystalsBatch(d_tempEvents.span());
        node::impl::d_simple_path_integral_batch(core::Image3DInput<float>{d_out_OSEMImg.grid, d_tmp1.get()},
                                                 d_tempEvents.cspan(), params.sample_rate, d_tempValues.get());
        node::impl::d_apply_correction_factor(d_tempValues, d_tempEvents.span(lors.size()), michNorm, michRand,
                                              michScat, michAttn, d_factors_add, d_factors_mul);
        node::impl::d_osem_fix_integral_value(d_tempValues.span(lors.size()));
        d_parallel_mul(d_tempValues.get(), d_tempMichValue.get(), d_tempValues.get(), lors.size());
        node::impl::d_fill_standard_events_values(d_tempEvents.get(), d_tempValues.get(), lors.size());
        node::impl::d_simple_path_reverse_integral_batch(core::Image3DOutput<float>{d_out_OSEMImg.grid, d_tmp2.get()},
                                                         d_tempEvents.cspan(lors.size()), params.sample_rate);
      }
      conv3D.deconvD(core::Image3DIO<float>{d_out_OSEMImg.grid, d_tmp2.get(), d_tmp2.get()});
      senmap.updateDImage(d_tmp2.get(), d_out_OSEMImg.ptr, subsetId, d_out_OSEMImg.grid);
    }
all_end:
}

void osem_impl_listmodeTOF_CUDA(
    core::Image3DOutput<float> d_out_OSEMImg, OSEM_TOF_params params, node::MichNormalization *michNorm,
    node::MichRandom *michRand, node::MichScatter *michScat, node::MichAttn *michAttn, core::MichDefine const &mich,
    OSEM_context_listmode &context) {
  // Process with listmode-specific logic using the converted LOR IDs
  const auto subsetPairs = tools::chunked_ranges_generator.by_group_count(0, context.events_count, params.subsetNum);
  const std::size_t batchSize = context.batchSize;

  openpni::cuda_sync_ptr<float> d_tempValues{"OSEM_tempValues"};
  cuda_sync_ptr<float> d_tempMichValue{"OSEM_tempMichValue"};
  cuda_sync_ptr<float> d_tmp1 = openpni::make_cuda_sync_ptr<float>(d_out_OSEMImg.grid.totalSize(), "OSEM_tmp1");
  cuda_sync_ptr<float> d_tmp2 = openpni::make_cuda_sync_ptr<float>(d_out_OSEMImg.grid.totalSize(), "OSEM_tmp2");
  cuda_sync_ptr<core::MichStandardEvent> d_tempEvents{"OSEM_tempEvents"};
  cuda_sync_ptr<float> d_factors_mul{"OSEM_MulFactor"};
  cuda_sync_ptr<float> d_factors_add{"OSEM_AddFactor"};

  auto &michCrystal = *context.michCrystal;
  auto &senmap = *context.senmap;
  auto &convolver = *context.convolver;

  for (const auto iteration : std::views::iota(0, params.iterNum))
    for (const auto subsetId : std::views::iota(0, params.subsetNum)) {
      PNI_DEBUG(std::format("OSEM iteration {}, subset {}\n", iteration, subsetId));

      convolver.convD(core::Image3DIO<float>{d_out_OSEMImg.grid, d_out_OSEMImg.ptr, d_tmp1.get()});
      d_tmp2.memset(0);

      const std::size_t subsetStart = subsetPairs[subsetId].first;
      const std::size_t subsetEnd = subsetPairs[subsetId].second;

      // Process this subset in batches
      for (std::size_t batchStart = subsetStart; batchStart < subsetEnd; batchStart += batchSize) {
        const std::size_t batchEnd = std::min(batchStart + batchSize, subsetEnd);
        const std::size_t currentBatchSize = batchEnd - batchStart;

        // Get events for this batch
        d_tempValues.reserve(currentBatchSize);
        d_tempMichValue.reserve(currentBatchSize);
        d_tempEvents.reserve(currentBatchSize);

        d_parallel_fill(d_tempMichValue.get(), 1.0f, currentBatchSize);
        node::impl::d_fill_standard_events_ids_from_listmodeTOF(
            d_tempEvents, context.Listmode_data.cspan().subspan(batchStart, currentBatchSize), params.TOF_division,
            mich);
        michCrystal.fillDCrystalsBatch(d_tempEvents.span());
        node::impl::d_simple_path_integral_batch_TOF(core::Image3DInput<float>{d_out_OSEMImg.grid, d_tmp1.get()},
                                                     d_tempEvents.cspan(), params.sample_rate, params.TOFBinWid_ps,
                                                     d_tempValues.get());
        // lgxtest
        // michRand->setRandomRatio(0.1);
        // PNI_DEBUG(std::format("RandomRatio: {}\n", michRand->getRandomRatio()));
        node::impl::d_apply_correction_factor(d_tempValues, d_tempEvents.span(currentBatchSize), michNorm, michRand,
                                              michScat, michAttn, d_factors_add, d_factors_mul);
        node::impl::d_osem_fix_integral_value(d_tempValues.span(currentBatchSize));
        d_parallel_mul(d_tempValues.get(), d_tempMichValue.get(), d_tempValues.get(), currentBatchSize);
        node::impl::d_fill_standard_events_values(d_tempEvents.get(), d_tempValues.get(), currentBatchSize);
        node::impl::d_simple_path_reverse_integral_batch_TOF(
            core::Image3DOutput<float>{d_out_OSEMImg.grid, d_tmp2.get()}, d_tempEvents.cspan(currentBatchSize),
            params.sample_rate, params.TOFBinWid_ps);
      }

      convolver.deconvD(core::Image3DIO<float>{d_out_OSEMImg.grid, d_tmp2.get(), d_tmp2.get()});
      senmap.updateDImage(d_tmp2.get(), d_out_OSEMImg.ptr, subsetId, d_out_OSEMImg.grid);
    }
all_end:
}

void backwardproj_impl_listmode_CUDA(
    core::Image3DOutput<float> d_out_OSEMImg, OSEM_TOF_params params, core::MichDefine const &mich,
    OSEM_context_listmode &context) {
  // Direct back-projection without forward projection (simple listmode back-projection)
  openpni::cuda_sync_ptr<float> d_tempValues{"BackProj_tempValues"};
  cuda_sync_ptr<float> d_tmp1 = openpni::make_cuda_sync_ptr<float>(d_out_OSEMImg.grid.totalSize(), "BackProj_tmp1");
  cuda_sync_ptr<core::MichStandardEvent> d_tempEvents{"BackProj_tempEvents"};

  auto &michCrystal = *context.michCrystal;
  auto &convolver = *context.convolver;

  const std::size_t batchSize = context.batchSize;

  d_tmp1.memset(0);

  // Process all events in batches for back-projection
  for (std::size_t batchStart = 0; batchStart < context.events_count; batchStart += batchSize) {
    const std::size_t batchEnd = std::min(batchStart + batchSize, context.events_count);
    const std::size_t currentBatchSize = batchEnd - batchStart;

    // Reserve memory for this batch
    d_tempValues.reserve(currentBatchSize);
    d_tempEvents.reserve(currentBatchSize);

    node::impl::d_fill_standard_events_ids_from_listmode(
        d_tempEvents, context.Listmode_data.cspan().subspan(batchStart, currentBatchSize), mich);

    michCrystal.fillDCrystalsBatch(d_tempEvents.span());

    // Set event values to 1.0 for simple back-projection (no normalization/correction applied)
    d_parallel_fill(d_tempValues.get(), 1.0f, currentBatchSize);
    node::impl::d_fill_standard_events_values(d_tempEvents.get(), d_tempValues.get(), currentBatchSize);

    // Direct back-projection: project values back to image space
    node::impl::d_simple_path_reverse_integral_batch(core::Image3DOutput<float>{d_out_OSEMImg.grid, d_tmp1.get()},
                                                     d_tempEvents.cspan(currentBatchSize), params.sample_rate);
  }

  // Apply convolution (deconvolution) to the accumulated back-projection result
  convolver.deconvD(core::Image3DIO<float>{d_out_OSEMImg.grid, d_tmp1.get(), d_out_OSEMImg.ptr});

  PNI_DEBUG("Direct back-projection complete.\n");
}
void osem_impl_listmode_CUDA(
    core::Image3DOutput<float> d_out_OSEMImg, OSEM_TOF_params params, node::MichNormalization *michNorm,
    node::MichRandom *michRand, node::MichScatter *michScat, node::MichAttn *michAttn, core::MichDefine const &mich,
    OSEM_context_listmode &context) {
  // Process with listmode-specific logic using the converted LOR IDs
  const auto subsetPairs = tools::chunked_ranges_generator.by_group_count(0, context.events_count, params.subsetNum);
  const std::size_t batchSize = context.batchSize;

  openpni::cuda_sync_ptr<float> d_tempValues{"OSEM_tempValues"};
  cuda_sync_ptr<float> d_tempMichValue{"OSEM_tempMichValue"};
  cuda_sync_ptr<float> d_tmp1 = openpni::make_cuda_sync_ptr<float>(d_out_OSEMImg.grid.totalSize(), "OSEM_tmp1");
  cuda_sync_ptr<float> d_tmp2 = openpni::make_cuda_sync_ptr<float>(d_out_OSEMImg.grid.totalSize(), "OSEM_tmp2");
  cuda_sync_ptr<core::MichStandardEvent> d_tempEvents{"OSEM_tempEvents"};
  cuda_sync_ptr<float> d_factors_mul{"OSEM_MulFactor"};
  cuda_sync_ptr<float> d_factors_add{"OSEM_AddFactor"};

  auto &michCrystal = *context.michCrystal;
  auto &senmap = *context.senmap;
  auto &convolver = *context.convolver;

  // test dump senmap
  auto senResult = senmap.dumpHSenmap(0, d_out_OSEMImg.grid);
  {
    std::ofstream senFile("/home/ustc/Desktop/testBi_dynamicCase/Data/result/Senmap_dump.bin", std::ios::binary);
    if (senFile.is_open()) {
      senFile.write(reinterpret_cast<const char *>(senResult.get()), d_out_OSEMImg.grid.totalSize() * sizeof(float));
      senFile.close();
      std::cout << "Senmap dumped to Data/result/Senmap_dump.bin" << std::endl;
    }
  }

  for (const auto iteration : std::views::iota(0, params.iterNum))
    for (const auto subsetId : std::views::iota(0, params.subsetNum)) {
      PNI_DEBUG(std::format("OSEM iteration {}, subset {}\n", iteration, subsetId));

      convolver.convD(core::Image3DIO<float>{d_out_OSEMImg.grid, d_out_OSEMImg.ptr, d_tmp1.get()});
      d_tmp2.memset(0);

      const std::size_t subsetStart = subsetPairs[subsetId].first;
      const std::size_t subsetEnd = subsetPairs[subsetId].second;

      // Process this subset in batches
      for (std::size_t batchStart = subsetStart; batchStart < subsetEnd; batchStart += batchSize) {
        const std::size_t batchEnd = std::min(batchStart + batchSize, subsetEnd);
        const std::size_t currentBatchSize = batchEnd - batchStart;

        // Get events for this batch
        d_tempValues.reserve(currentBatchSize);
        d_tempMichValue.reserve(currentBatchSize);
        d_tempEvents.reserve(currentBatchSize);

        d_parallel_fill(d_tempMichValue.get(), 1.0f, currentBatchSize);

        node::impl::d_fill_standard_events_ids_from_listmode(
            d_tempEvents, context.Listmode_data.cspan().subspan(batchStart, currentBatchSize), mich);

        michCrystal.fillDCrystalsBatch(d_tempEvents.span());

        node::impl::d_simple_path_integral_batch(core::Image3DInput<float>{d_out_OSEMImg.grid, d_tmp1.get()},
                                                 d_tempEvents.cspan(), params.sample_rate, d_tempValues.get());
        node::impl::d_apply_correction_factor(d_tempValues, d_tempEvents.span(currentBatchSize), michNorm, michRand,
                                              michScat, michAttn, d_factors_add, d_factors_mul);
        node::impl::d_osem_fix_integral_value(d_tempValues.span(currentBatchSize));
        d_parallel_mul(d_tempValues.get(), d_tempMichValue.get(), d_tempValues.get(), currentBatchSize);
        node::impl::d_fill_standard_events_values(d_tempEvents.get(), d_tempValues.get(), currentBatchSize);
        node::impl::d_simple_path_reverse_integral_batch(core::Image3DOutput<float>{d_out_OSEMImg.grid, d_tmp2.get()},
                                                         d_tempEvents.cspan(currentBatchSize), params.sample_rate);
      }

      convolver.deconvD(core::Image3DIO<float>{d_out_OSEMImg.grid, d_tmp2.get(), d_tmp2.get()});
      senmap.updateDImage(d_tmp2.get(), d_out_OSEMImg.ptr, subsetId, d_out_OSEMImg.grid);
    }
all_end:
}

void instant_OSEM_mich_CUDA(
    core::Image3DOutput<float> out_OSEMImg, OSEM_params params, interface::Conv3D &conv3D, float const *michValue,
    node::MichNormalization *michNorm, node::MichRandom *michRand, node::MichScatter *michScat,
    node::MichAttn *michAttn, core::MichDefine const &mich) {
  if (!michScat)
    params.scatterSimulations = 0;

  auto [grid, h_outImg] = out_OSEMImg;
  auto d_out_OSEMImg = make_cuda_sync_ptr<float>(grid.totalSize(), "OSEM_outImg");

  OSEM_context context;
  context.lorBatch = std::make_unique<node::LORBatch>(mich);
  context.lorBatch->setSubsetNum(params.subsetNum).setBinCut(params.binCutRatio * MichInfoHub(mich).getBinNum());
  context.michCrystal = std::make_unique<node::MichCrystal>(mich);
  context.senmap = std::make_unique<node::MichSenmap>(conv3D, mich);
  context.senmap->setSubsetNum(params.subsetNum);
  if (michNorm)
    context.senmap->bindNormalization(michNorm);
  if (michAttn)
    context.senmap->bindAttenuation(michAttn);
  context.d_michValue = openpni::make_cuda_sync_ptr_from_hcopy(
      std::span<float const>{michValue, MichInfoHub(mich).getMichSize()}, "OSEM_michValue");
  if (michScat)
    michScat->bindDPromptMich(context.d_michValue);

  for (auto scatterIter = 0; scatterIter <= params.scatterSimulations; scatterIter++) {
    d_parallel_fill(d_out_OSEMImg.get(), 1.0f, grid.totalSize());
    osem_impl_CUDA(core::Image3DOutput<float>{grid, d_out_OSEMImg}, params, conv3D, michNorm, michRand, michScat,
                   michAttn, mich, context);
    if (michScat)
      michScat->bindDEmissionMap(grid, d_out_OSEMImg);
  }

  d_out_OSEMImg.allocator().copy_from_device_to_host(h_outImg, d_out_OSEMImg.cspan());
}

struct DelayListmodeReadResult {
  std::size_t eventCount = 0;
  std::unique_ptr<node::MichRandom> michRand;
};

DelayListmodeReadResult read_delay_listmode(
    node::MichRandom *existingMichRand, std::string const &randomListmodeFile, uint32_t listmodeFileTimeBegin_ms,
    uint32_t listmodeFileTimeEnd_ms, core::MichDefine const &mich) {
  DelayListmodeReadResult result;
  if (randomListmodeFile.size() && existingMichRand) {
    result.michRand = existingMichRand->copyPtr();
    openpni::io::ListmodeFileInput delayListmodeFile;
    delayListmodeFile.open(randomListmodeFile);
    auto selectedDelayListmodeSegments =
        openpni::io::selectSegments(delayListmodeFile, listmodeFileTimeBegin_ms, listmodeFileTimeEnd_ms);
    auto totalDelayEvents = std::accumulate(selectedDelayListmodeSegments.begin(), selectedDelayListmodeSegments.end(),
                                            0ull, [](auto a, auto b) { return a + b.dataIndexEnd - b.dataIndexBegin; });
    PNI_DEBUG("Delay Listmode file opened, reading segments...");

    openpni::misc::ListmodeBuffer delayListmodeBuffer;
    delayListmodeBuffer
        .setBufferSize(1024 * 1024 * 1000) // 1000M
        .callWhenFlush([&](const openpni::basic::Listmode_t *__data, std::size_t __count) {
          PNI_DEBUG(std::format("Processing {} delay listmode events...\n", __count));
          result.michRand->addDelayListmodes(std::span<const openpni::basic::Listmode_t>(__data, __count));
        })
        .append(delayListmodeFile, selectedDelayListmodeSegments)
        .flush();

    result.eventCount = totalDelayEvents;
  }
  return result;
}

DelayListmodeReadResult read_delay_listmode(
    std::vector<std::string> const &randomListmodeFiles, uint32_t listmodeFileTimeBegin_ms,
    uint32_t listmodeFileTimeEnd_ms, int minSector, int radialModeulNumS, core::MichDefine const &mich) {
  DelayListmodeReadResult result;
  for (const auto &randomListmodeFile : randomListmodeFiles) {
    if (!randomListmodeFile.size())
      continue;

    if (!result.michRand) {
      result.michRand = std::make_unique<node::MichRandom>(mich);
      result.michRand->setMinSectorDifference(minSector);
      result.michRand->setRadialModuleNumS(radialModeulNumS);
      result.michRand->addDelayListmodes({});
    }
    openpni::io::ListmodeFileInput delayListmodeFile;
    delayListmodeFile.open(randomListmodeFile);

    auto selectedDelayListmodeSegments =
        openpni::io::selectSegments(delayListmodeFile, listmodeFileTimeBegin_ms, listmodeFileTimeEnd_ms);
    auto totalDelayEvents = std::accumulate(selectedDelayListmodeSegments.begin(), selectedDelayListmodeSegments.end(),
                                            0ull, [](auto a, auto b) { return a + b.dataIndexEnd - b.dataIndexBegin; });
    PNI_DEBUG(std::format("Delay Listmode file {} opened, reading segments... Total events: {}, segments: {}\n",
                          randomListmodeFile, totalDelayEvents, selectedDelayListmodeSegments.size()));
    openpni::misc::ListmodeBuffer delayListmodeBuffer;
    delayListmodeBuffer
        .setBufferSize(1024 * 1024 * 1024) // 1000M
        .callWhenFlush([&](const openpni::basic::Listmode_t *__data, std::size_t __count) {
          PNI_DEBUG(std::format("Processing {} delay listmode events...\n", __count));
          result.michRand->addDelayListmodes(std::span<const openpni::basic::Listmode_t>(__data, __count));
        })
        .append(delayListmodeFile, selectedDelayListmodeSegments)
        .flush();

    result.eventCount = totalDelayEvents;
  }
  return result;
}
void instant_backwardProjection_listmode_CUDA(
    core::Image3DOutput<float> out_Img, OSEM_TOF_params params, interface::Conv3D &conv3D, std::string listmode_path,
    core::MichDefine const &mich) {
  auto [grid, h_outImg] = out_Img;
  auto d_out_OSEMImg = make_cuda_sync_ptr<float>(grid.totalSize(), "backProj_outImg");
  // read listmode
  openpni::io::ListmodeFileInput listmodeFile;
  listmodeFile.open(listmode_path);
  auto selectedListmodeSegments =
      openpni::io::selectSegments(listmodeFile, params.listmodeFileTimeBegin_ms, params.listmodeFileTimeEnd_ms);
  auto totalEvents = std::accumulate(selectedListmodeSegments.begin(), selectedListmodeSegments.end(), 0ull,
                                     [](auto a, auto b) { return a + b.dataIndexEnd - b.dataIndexBegin; });
  PNI_DEBUG(std::format("Listmode file opened, reading segments... Total events: {}\n", totalEvents));

  OSEM_context_listmode context;
  context.michCrystal = std::make_unique<node::MichCrystal>(mich);
  context.convolver = &conv3D;

  // Prepare listmode buffer for chunked reading
  openpni::misc::ListmodeBuffer listmodeBuffer;

  auto GBSize = [](unsigned long long size) -> uint64_t { return size * 1024 * 1024 * 1024; };

  PNI_DEBUG(std::format("Setting up listmode buffer with {} GB...\n", params.size_GB));

  listmodeBuffer.setBufferSize(GBSize(params.size_GB) / sizeof(openpni::basic::Listmode_t))
      .callWhenFlush([&](const openpni::basic::Listmode_t *__data, std::size_t __count) {
        PNI_DEBUG(std::format("Total events: {}, processing {} events in this chunk.\n", totalEvents, __count));
        context.events_count = __count;

        // Copy listmode data to device
        if (context.Listmode_data.elements() < __count)
          context.Listmode_data = openpni::make_cuda_sync_ptr<openpni::basic::Listmode_t>(__count, "ListmodeBuffer");

        context.Listmode_data.allocator().copy_from_host_to_device(
            context.Listmode_data.get(), std::span<const openpni::basic::Listmode_t>(__data, __count));

        backwardproj_impl_listmode_CUDA(core::Image3DOutput<float>{grid, d_out_OSEMImg}, params, mich, context);
        PNI_DEBUG("Chunk processing complete.\n");
      })
      .append(listmodeFile,
              openpni::io::selectSegments(listmodeFile, params.listmodeFileTimeBegin_ms, params.listmodeFileTimeEnd_ms))
      .flush();

  d_out_OSEMImg.allocator().copy_from_device_to_host(h_outImg, d_out_OSEMImg.cspan());
  PNI_DEBUG("listmode backward projection complete.\n");
}

void instant_OSEM_listmode_CUDA(
    core::Image3DOutput<float> out_OSEMImg, OSEM_TOF_params params, interface::Conv3D &conv3D,
    std::string listmode_path, node::MichNormalization *michNorm, node::MichRandom *__michRand,
    node::MichScatter *michScat, node::MichAttn *michAttn, core::MichDefine const &mich) {
  if (!michScat)
    params.scatterSimulations = 0;

  auto [grid, h_outImg] = out_OSEMImg;
  auto d_out_OSEMImg = make_cuda_sync_ptr<float>(grid.totalSize(), "OSEM_outImg");

  // read listmode
  openpni::io::ListmodeFileInput listmodeFile;
  listmodeFile.open(listmode_path);
  auto selectedListmodeSegments =
      openpni::io::selectSegments(listmodeFile, params.listmodeFileTimeBegin_ms, params.listmodeFileTimeEnd_ms);
  auto totalEvents = std::accumulate(selectedListmodeSegments.begin(), selectedListmodeSegments.end(), 0ull,
                                     [](auto a, auto b) { return a + b.dataIndexEnd - b.dataIndexBegin; });
  PNI_DEBUG(std::format("Listmode file opened, reading segments... Total events: {}\n", totalEvents));

  OSEM_context_listmode context;
  context.michCrystal = std::make_unique<node::MichCrystal>(mich);
  context.senmap = std::make_unique<node::MichSenmap>(conv3D, mich);
  context.senmap->setSubsetNum(params.subsetNum);
  context.senmap->setMode(node::MichSenmap::Mode_listmode);
  if (michNorm)
    context.senmap->bindNormalization(michNorm);
  if (michAttn)
    context.senmap->bindAttenuation(michAttn);
  context.convolver = &conv3D;

  auto [delayListmodeCount, michRand] = read_delay_listmode(
      __michRand, params.randomListmodeFile, params.listmodeFileTimeBegin_ms, params.listmodeFileTimeEnd_ms, mich);

  // Prepare listmode buffer for chunked reading
  openpni::misc::ListmodeBuffer listmodeBuffer;

  auto GBSize = [](unsigned long long size) -> uint64_t { return size * 1024 * 1024 * 1024; };

  PNI_DEBUG(std::format("Setting up listmode buffer with {} GB...\n", params.size_GB));

  listmodeBuffer.setBufferSize(GBSize(params.size_GB) / sizeof(openpni::basic::Listmode_t))
      .callWhenFlush([&](const openpni::basic::Listmode_t *__data, std::size_t __count) {
        PNI_DEBUG(std::format("Total events: {}, processing {} events in this chunk.\n", totalEvents, __count));
        context.events_count = __count;

        // Copy listmode data to device
        if (context.Listmode_data.elements() < __count)
          context.Listmode_data = openpni::make_cuda_sync_ptr<openpni::basic::Listmode_t>(__count, "ListmodeBuffer");

        context.Listmode_data.allocator().copy_from_host_to_device(
            context.Listmode_data.get(), std::span<const openpni::basic::Listmode_t>(__data, __count));

        if (michScat)
          michScat->bindDListmode(context.Listmode_data.cspan(__count));
        if (michRand)
          michRand->setCountRatio(double(__count) / double(totalEvents));
        for (auto scatterIter = 0; scatterIter <= params.scatterSimulations; scatterIter++) {
          d_parallel_fill(d_out_OSEMImg.get(), 1.0f, grid.totalSize());
          osem_impl_listmode_CUDA(core::Image3DOutput<float>{grid, d_out_OSEMImg}, params, michNorm, michRand.get(),
                                  michScat, michAttn, mich, context);
          if (michScat && scatterIter < params.scatterSimulations)
            michScat->bindDEmissionMap(grid, d_out_OSEMImg);
        }

        PNI_DEBUG("Chunk processing complete.\n");
      })
      .append(listmodeFile,
              openpni::io::selectSegments(listmodeFile, params.listmodeFileTimeBegin_ms, params.listmodeFileTimeEnd_ms))
      .flush();

  d_out_OSEMImg.allocator().copy_from_device_to_host(h_outImg, d_out_OSEMImg.cspan());
  PNI_DEBUG("OSEM listmode reconstruction complete.\n");
}

void instant_OSEM_listmodeTOF_CUDA(
    core::Image3DOutput<float> out_OSEMImg, OSEM_TOF_params params, interface::Conv3D &conv3D,
    std::string listmode_path, node::MichNormalization *michNorm, node::MichRandom *__michRand,
    node::MichScatter *michScat, node::MichAttn *michAttn, core::MichDefine const &mich) {
  if (!michScat)
    params.scatterSimulations = 0;

  auto [grid, h_outImg] = out_OSEMImg;
  auto d_out_OSEMImg = make_cuda_sync_ptr<float>(grid.totalSize(), "OSEM_outImg");

  // read listmode
  openpni::io::ListmodeFileInput listmodeFile;
  listmodeFile.open(listmode_path);
  auto selectedListmodeSegments =
      openpni::io::selectSegments(listmodeFile, params.listmodeFileTimeBegin_ms, params.listmodeFileTimeEnd_ms);
  auto totalEvents = std::accumulate(selectedListmodeSegments.begin(), selectedListmodeSegments.end(), 0ull,
                                     [](auto a, auto b) { return a + b.dataIndexEnd - b.dataIndexBegin; });
  PNI_DEBUG(std::format("Listmode file opened, reading segments... Total events: {}\n", totalEvents));

  OSEM_context_listmode context;
  context.michCrystal = std::make_unique<node::MichCrystal>(mich);
  context.senmap = std::make_unique<node::MichSenmap>(conv3D, mich);
  context.senmap->setSubsetNum(params.subsetNum);
  context.senmap->setMode(node::MichSenmap::Mode_listmode);
  if (michNorm)
    context.senmap->bindNormalization(michNorm);
  if (michAttn)
    context.senmap->bindAttenuation(michAttn);
  context.convolver = &conv3D;

  auto [delayListmodeCount, michRand] = read_delay_listmode(
      __michRand, params.randomListmodeFile, params.listmodeFileTimeBegin_ms, params.listmodeFileTimeEnd_ms, mich);

  // Prepare listmode buffer for chunked reading
  openpni::misc::ListmodeBuffer listmodeBuffer;

  PNI_DEBUG(std::format("Setting up listmode buffer with {} GB...\n", params.size_GB));

  listmodeBuffer.setBufferSize(_GB(params.size_GB) / sizeof(openpni::basic::Listmode_t))
      .callWhenFlush([&](const openpni::basic::Listmode_t *__data, std::size_t __count) {
        PNI_DEBUG(std::format("Total events: {}, processing {} events in this chunk.\n", totalEvents, __count));
        context.events_count = __count;

        // Copy listmode data to device
        if (context.Listmode_data.elements() < __count)
          context.Listmode_data = openpni::make_cuda_sync_ptr<openpni::basic::Listmode_t>(__count, "ListmodeBuffer");

        context.Listmode_data.allocator().copy_from_host_to_device(
            context.Listmode_data.get(), std::span<const openpni::basic::Listmode_t>(__data, __count));

        if (michScat)
          michScat->bindDListmode(context.Listmode_data.cspan(__count));
        if (michRand) {
          michRand->setCountRatio(double(__count) / double(totalEvents));
          michRand->setTimeBinRatio(double(params.TOFBinWid_ps) / double(params.timeWindow_ps));
        }

        for (auto scatterIter = 0; scatterIter <= params.scatterSimulations; scatterIter++) {
          d_parallel_fill(d_out_OSEMImg.get(), 1.0f, grid.totalSize());
          osem_impl_listmodeTOF_CUDA(core::Image3DOutput<float>{grid, d_out_OSEMImg}, params, michNorm, michRand.get(),
                                     michScat, michAttn, mich, context);
          if (michScat && scatterIter < params.scatterSimulations)
            michScat->bindDEmissionMap(grid, d_out_OSEMImg);
        }

        PNI_DEBUG("Chunk processing complete.\n");
      })
      .append(listmodeFile,
              openpni::io::selectSegments(listmodeFile, params.listmodeFileTimeBegin_ms, params.listmodeFileTimeEnd_ms))
      .flush();

  d_out_OSEMImg.allocator().copy_from_device_to_host(h_outImg, d_out_OSEMImg.cspan());
  PNI_DEBUG("OSEM listmode reconstruction complete.\n");
}
std::vector<int> get_gpu_id(
    uint16_t bitmap_gpu_usage) {
  std::vector<int> gpu_ids;
  int gpuNum = 0;
  cudaGetDeviceCount(&gpuNum);
  for (int i = 0; i < std::min(gpuNum, 16); ++i) {
    if (bitmap_gpu_usage & (1 << i)) {
      gpu_ids.push_back(i);
    }
  }
  return gpu_ids;
}

void instant_OSEM_listmodeTOF_MULTI_CUDA(
    core::Image3DOutput<float> out_OSEMImg, OSEM_TOF_MULTI_params params, core::MichDefine const &mich) {
  if (params.listmode_paths.empty())
    throw std::runtime_error("No listmode files provided.");

  if (!params.doScatter)
    params.scatterSimulations = 0;

  auto [grid, h_outImg] = out_OSEMImg;

  struct ListmodeFile {
    std::unique_ptr<openpni::io::ListmodeFileInput> file;
    std::vector<openpni::io::_SegmentView> segments;
    std::size_t startTime_ms = 0xffffffff;
    std::size_t totalEventsOfFile = 0;
  };
  std::vector<ListmodeFile> listmodeFiles;
  for (const auto &path : params.listmode_paths) {
    listmodeFiles.emplace_back();
    auto &lmFile = listmodeFiles.back();
    lmFile.file = std::make_unique<openpni::io::ListmodeFileInput>();
    lmFile.file->open(path);
    for (const auto headerIndex : std::views::iota(0u, lmFile.file->segmentNum())) {
      lmFile.startTime_ms = std::min(lmFile.startTime_ms, lmFile.file->segmentHeader(headerIndex).clock);
    }
  }
  const auto timeOffset_ms =
      std::min_element(listmodeFiles.begin(), listmodeFiles.end(), [](auto const &a, auto const &b) {
        return a.startTime_ms < b.startTime_ms;
      })->startTime_ms;
  for (auto &lmFile : listmodeFiles) {
    lmFile.segments = openpni::io::selectSegments(*lmFile.file, params.timeBegin_ms + timeOffset_ms,
                                                  params.timeEnd_ms + timeOffset_ms);
    lmFile.totalEventsOfFile = std::accumulate(lmFile.segments.begin(), lmFile.segments.end(), 0ull,
                                               [](auto a, auto &b) { return a + b.dataIndexEnd - b.dataIndexBegin; });
  }

  auto [delayListmodeCount, michRand] =
      read_delay_listmode(params.randomListmodeFile, params.timeBegin_ms + timeOffset_ms,
                          params.timeEnd_ms + timeOffset_ms, params.minSectorDiff, params.randRadialModuleNumS, mich);

  const auto totalEvents = std::accumulate(listmodeFiles.begin(), listmodeFiles.end(), 0ull,
                                           [](auto a, auto &b) { return a + b.totalEventsOfFile; });
  PNI_DEBUG(std::format("Listmode file opened, reading segments... Total events: {}\n", totalEvents));

  struct MultiGPUData {
    std::vector<openpni::basic::Listmode_t> listmodes;
  };
  const auto gpu_ids = get_gpu_id(params.bitmap_gpu_usage);

  PNI_DEBUG(std::format("System GPU num = {}\n", gpu_ids.size()));
  if (gpu_ids.empty())
    throw std::runtime_error("No CUDA device found.");

  common::MultiCycledBuffer<MultiGPUData> cycleBuffer(gpu_ids.size());
  std::vector<std::thread> threadsGPU;
  std::mutex h_addMutex;
  for (const auto &gpu_id : gpu_ids) {
    threadsGPU.emplace_back([&, gpu_id]() {
      cudaSetDevice(gpu_id);

      auto norm = params.doNorm ? std::make_unique<node::MichNormalization>(mich) : nullptr;
      if (norm) {
        norm->recoverFromFile(params.normFactorsFile);
        if (params.doSelfNorm)
          norm->bindSelfNormMich(params.selfNormMich);
        if (params.doDeadTime)
          norm->setDeadTimeTable(params.deadTimetable);
      }

      auto attn = params.doAttn ? std::make_unique<node::MichAttn>(mich) : nullptr;
      if (attn) {
        attn->setFetchMode(node::MichAttn::FromPreBaked);
        attn->setPreferredSource(node::MichAttn::Attn_GPU);
        attn->setMapSize(params.attnMap.grid);
        attn->bindHAttnMap(params.attnMap.ptr);
      }

      auto rand = michRand ? michRand->copyPtr() : std::unique_ptr<node::MichRandom>{};

      auto scat = (params.doScatter && rand && attn && norm) ? std::make_unique<node::MichScatter>(mich) : nullptr;
      if (scat) {
        scat->setMinSectorDifference(params.minSectorDiff);
        scat->setTailFittingThreshold(params.tailFittingThreshold);
        scat->setScatterPointsThreshold(params.scatterPointsThreshold);
        scat->setScatterEnergyWindow(params.scatterEnergyWindow);
        scat->setScatterEffTableEnergy(params.scatterEffTableEnergy);
        scat->setTOFParams(params.tofSSS_timeBinWidth_ns, params.tofSSS_timeBinStart_ns, params.tofSSS_timeBinEnd_ns,
                           params.tofSSS_systemTimeRes_ns);
        scat->bindAttnCoff(attn.get());
        scat->bindNorm(norm.get());
        scat->bindRandom(rand.get());
        scat->setScatterPointGrid(params.scatterPointGrid);
      }

      // node::GaussianConv3D conv3D;
      // conv3D.setHWHM(params.gauss_hwhm_mm);
      //
      std::unique_ptr<interface::Conv3D> convPtr;
      //
      if (std::holds_alternative<GaussianConvParams>(params.convParams)) {
        auto gaussConv3D = std::make_unique<node::GaussianConv3D>();
        auto convParams = std::get<GaussianConvParams>(params.convParams);
        gaussConv3D->setHWHM(convParams.HWHM);
        convPtr = std::move(gaussConv3D);
      } else if (std::holds_alternative<KNNConvParams>(params.convParams)) {
        auto knnConv3D = std::make_unique<node::KNNConv3D>();
        auto convParams = std::get<KNNConvParams>(params.convParams);
        knnConv3D->setKNNNumbers(convParams.KNNNumbers);
        knnConv3D->setFeatureSizeHalf(convParams.FeatureSizeHalf);
        knnConv3D->setKNNSearchSizeHalf(convParams.SearchSizeHalf);
        knnConv3D->setKNNSigmaG2(convParams.SigmaG2);
        convPtr = std::move(knnConv3D);
      } else if (std::holds_alternative<GKNNConvParams>(params.convParams)) {
        auto convParams = std::get<GKNNConvParams>(params.convParams);
        auto d_kemImgData =
            make_cuda_sync_ptr_from_hcopy(std::span<const float>(convParams.h_kemImgTensorDataIn.ptr,
                                                                 convParams.h_kemImgTensorDataIn.grid.totalSize()),
                                          "GKNN_kemImgTensorDataIn");
        auto kemGrids = convParams.h_kemImgTensorDataIn.grid;
        auto gknnConv3D = std::make_unique<node::KGConv3D>();
        gknnConv3D->setKNNNumbers(convParams.KNNNumbers);
        gknnConv3D->setFeatureSizeHalf(convParams.FeatureSizeHalf);
        gknnConv3D->setKNNSearchSizeHalf(convParams.SearchSizeHalf);
        gknnConv3D->setKNNSigmaG2(convParams.SigmaG2);
        gknnConv3D->setHWHM(convParams.HWHM);
        gknnConv3D->setDTensorDataIn(d_kemImgData.get(), kemGrids);
        convPtr = std::move(gknnConv3D);
        convPtr->convD(core::Image3DIO<float>{kemGrids, d_kemImgData.get(), d_kemImgData.get()});

        // gknnConv3D->setKNNNumbers(std::get<GKNNConvParams>(params.convParams).KNNNumbers);
        // gknnConv3D->setFeatureSizeHalf(std::get<GKNNConvParams>(params.convParams).FeatureSizeHalf);
        // gknnConv3D->setKNNSearchSizeHalf(std::get<GKNNConvParams>(params.convParams).SearchSizeHalf);
        // gknnConv3D->setKNNSigmaG2(std::get<GKNNConvParams>(params.convParams).SigmaG2);
        // gknnConv3D->setHWHM(std::get<GKNNConvParams>(params.convParams).HWHM);
        // gknnConv3D->setDTensorDataIn((std::get<GKNNConvParams>(params.convParams)).getDTensorDataIn());
        // convPtr = std::move(gknnConv3D);
        // set bigVoxelScatterSimulation to false to avoid issues
        params.bigVoxelScatterSimulation = false;

        // // test to see if gknn set correctly
        // auto d_kemTestImgData = make_cuda_sync_ptr_from_hcopy(
        //     std::span<const float>(std::get<GKNNConvParams>(params.convParams).h_kemImgTensorDataIn.ptr,
        //                            std::get<GKNNConvParams>(params.convParams).h_kemImgTensorDataIn.grid.totalSize()),
        //     "GKNN_kemImgTestTensorDataIn");
        // convPtr->convD(core::Image3DIO<float>{kemGrids, d_kemTestImgData.get(), d_kemTestImgData.get()});
        // convPtr->deconvD(core::Image3DIO<float>{kemGrids, d_kemTestImgData.get(), d_kemTestImgData.get()});
        // // save
        // {
        //   std::vector<float> h_out_testKemImg(kemGrids.totalSize());
        //   d_kemTestImgData.allocator().copy_from_device_to_host(h_out_testKemImg.data(), d_kemTestImgData.cspan());
        //   std::ofstream orginImgFile("/media/cmx/K1/v4_coin/data/930TOFOSEMTest/Original_KEMTEST.bin",
        //                              std::ios::binary);
        //   if (orginImgFile.is_open()) {
        //     orginImgFile.write(
        //         reinterpret_cast<const char *>(std::get<GKNNConvParams>(params.convParams).h_kemImgTensorDataIn.ptr),
        //         kemGrids.totalSize() * sizeof(float));
        //     orginImgFile.close();
        //     std::cout << "Original_KEMTEST dumped to Data/result/Original_KEMTEST.bin" << std::endl;
        //   }

        //   std::ofstream kemFile("/media/cmx/K1/v4_coin/data/930TOFOSEMTest/KEMTEST.bin", std::ios::binary);
        //   if (kemFile.is_open()) {
        //     kemFile.write(reinterpret_cast<const char *>(h_out_testKemImg.data()),
        //                   kemGrids.totalSize() * sizeof(float));
        //     kemFile.close();
        //     std::cout << "GKNN_kemImgTensorDataOut dumped to Data/result/GKNN_kemImgTensorDataOut.bin" << std::endl;
        //   }
        // }
      }
      OSEM_context_listmode context;
      context.michCrystal = std::make_unique<node::MichCrystal>(mich);
      context.senmap = std::make_unique<node::MichSenmap>(*convPtr, mich);
      context.senmap->setSubsetNum(params.subsetNum);
      context.senmap->setMode(node::MichSenmap::Mode_listmode);
      if (norm)
        context.senmap->bindNormalization(norm.get());
      if (attn)
        context.senmap->bindAttenuation(attn.get());
      context.convolver = convPtr.get();

      PNI_DEBUG(std::format("Thread for GPU {} started.\n", gpu_id));
      while (cycleBuffer.read([&](const MultiGPUData &data) {
        PNI_DEBUG(std::format("Processing {} listmode events...\n", data.listmodes.size()));
        context.events_count = data.listmodes.size();

        auto d_out_OSEMImg = make_cuda_sync_ptr<float>(grid.totalSize(), "OSEM_outImg");
        auto d_out_OSEMImgPair = make_cuda_sync_ptr<float>(grid.totalSize(), "OSEM_outImgPair");
        auto bigGrid = core::Grids<3>::create_by_center_boxLength_size(
            grid.center(), grid.boxLength(), (grid.size.dimSize * 0.6).template to<int64_t>());

        // Copy listmode data to device
        context.Listmode_data.reserve(data.listmodes.size());
        context.Listmode_data.allocator().copy_from_host_to_device(
            context.Listmode_data.get(), std::span<openpni::basic::Listmode_t const>(data.listmodes));

        if (scat) {
          scat->bindDListmode(context.Listmode_data.span(context.events_count));
          scat->setTOFParams(params.tofSSS_timeBinWidth_ns, params.tofSSS_timeBinStart_ns, params.tofSSS_timeBinEnd_ns,
                             params.tofSSS_systemTimeRes_ns);
        }
        if (rand) {
          auto countRatio = double(context.events_count) / double(totalEvents);
          auto timeRatio = double(params.TOFBinWid_ps) / double(params.timeWindow_ps);
          michRand->setCountRatio(countRatio);
          michRand->setTimeBinRatio(timeRatio);
          PNI_DEBUG(std::format("GPU {} set count ratio to {},timeBin ratio to {} ,events_count {} total {}\n", gpu_id,
                                countRatio, timeRatio, context.events_count, totalEvents));
          // // test
          // auto randresult = rand->dumpFactorsAsHMich();
          // // save
          // {
          //   auto michInfo = MichInfoHub(mich);

          //   std::ofstream randFile(
          //       std::format("/media/cmx/K1/v4_coin/data/930TOFOSEMTest/RandomFactors_GPU{}.bin", gpu_id),
          //       std::ios::binary);
          //   if (randFile.is_open()) {
          //     randFile.write(reinterpret_cast<const char *>(randresult.get()), michInfo.getMichSize() *
          //     sizeof(float)); randFile.close(); std::cout << std::format("RandomFactors_GPU{} dumped to
          //     Data/result/RandomFactors_GPU{}.bin", gpu_id,
          //                              gpu_id)
          //               << std::endl;
          //   }
          // }
        }

        auto *t_norm = norm.get();
        auto *t_rand = rand.get();
        auto *t_scat = scat.get();
        auto *t_attn = attn.get();

        if (t_scat)
          t_scat->bindDListmode(context.Listmode_data.span(context.events_count));

        OSEM_TOF_params _params;
        _params.iterNum = params.iterNum;
        _params.subsetNum = params.subsetNum;
        _params.sample_rate = params.sample_rate;
        _params.TOF_division = params.TOF_division_ps;
        _params.binCutRatio = params.binCutRatio;
        _params.scatterSimulations = params.scatterSimulations;
        _params.listmodeFileTimeBegin_ms = params.timeBegin_ms;
        _params.listmodeFileTimeEnd_ms = params.timeEnd_ms;

        if (!scat)
          params.scatterSimulations = 0;

        for (auto scatterIter = 0; scatterIter <= params.scatterSimulations; scatterIter++) {
          auto osem_grid = grid;
          if (scat && scatterIter < params.scatterSimulations && params.bigVoxelScatterSimulation)
            osem_grid = bigGrid;
          d_parallel_fill(d_out_OSEMImg.get(), 1.0f, grid.totalSize());
          osem_impl_listmodeTOF_CUDA(core::Image3DOutput<float>{osem_grid, d_out_OSEMImg}, _params, t_norm, t_rand,
                                     t_scat, t_attn, mich, context);
          d_out_OSEMImg.allocator().copy_from_device_to_device(d_out_OSEMImgPair.data(), d_out_OSEMImg.cspan());
          // if KEM or GKEM method,deconvelution after reconstruction
          if (std::holds_alternative<GKNNConvParams>(params.convParams) ||
              std::holds_alternative<GaussianConvParams>(params.convParams)) {
            convPtr->deconvD(core::Image3DIO<float>{osem_grid, d_out_OSEMImgPair.get(), d_out_OSEMImgPair.get()});
          }
          if (scat && scatterIter < params.scatterSimulations)
            scat->bindDEmissionMap(osem_grid, d_out_OSEMImgPair);
        }

        PNI_DEBUG(std::format("Chunk processing complete.\n"));
        // if knn do one more conv
        if (std::holds_alternative<GKNNConvParams>(params.convParams) ||
            std::holds_alternative<KNNConvParams>(params.convParams)) {
          convPtr->deconvD(core::Image3DIO<float>{grid, d_out_OSEMImgPair.get(), d_out_OSEMImgPair.get()});
        }

        auto h_img = make_vector_from_cuda_sync_ptr(d_out_OSEMImg, d_out_OSEMImg.cspan());
        std::lock_guard lock(h_addMutex);
        tools::parallel_for_each(d_out_OSEMImg.elements(), [&](size_t idx) { out_OSEMImg.ptr[idx] += h_img[idx]; });
        PNI_DEBUG(std::format("GPU {} image max value: {}.\n", gpu_id,
                              *std::max_element(h_img.data(), h_img.data() + grid.totalSize())));
      }))
        ;
      PNI_DEBUG(std::format("Thread for GPU {} ending.\n", gpu_id));
    });
  }
  misc::ListmodeBuffer listmodeBuffer;
  listmodeBuffer
      .setBufferSize(find_value(totalEvents, _GB(params.size_GB) / sizeof(openpni::basic::Listmode_t), gpu_ids.size()))
      .callWhenFlush([&](const openpni::basic::Listmode_t *__data, std::size_t __count) noexcept {
        cycleBuffer.write([&](MultiGPUData &data) noexcept {
          data.listmodes.resize(__count);
          std::memcpy(data.listmodes.data(), __data, __count * sizeof(openpni::basic::Listmode_t));
          PNI_DEBUG(std::format("copy listmode to multiGPUData done, count = {}\n", __count));
        });
      });
  for (auto &lmFile : listmodeFiles)
    listmodeBuffer.append(*lmFile.file, lmFile.segments);
  listmodeBuffer.flush();
  cycleBuffer.stop();

  for (auto &t : threadsGPU)
    t.join();
}

void instant_FBP_mich_CUDA(
    core::Image3DOutput<float> out_FBPImg, FBP_params params, float const *h_michValue, core::MichDefine const &mich) {
  // 将 experimental::core::PolygonalSystem 转换为 example::PolygonalSystem
  openpni::example::PolygonalSystem examplePolygon;
  examplePolygon.edges = mich.polygon.edges;
  examplePolygon.detectorPerEdge = mich.polygon.detectorPerEdge;
  examplePolygon.detectorLen = mich.polygon.detectorLen;
  examplePolygon.radius = mich.polygon.radius;
  examplePolygon.angleOf1stPerp = mich.polygon.angleOf1stPerp;
  examplePolygon.detectorRings = mich.polygon.detectorRings;
  examplePolygon.ringDistance = mich.polygon.ringDistance;

  auto e180Builder = openpni::example::polygon::PolygonModelBuilder<openpni::device::bdm2::BDM2Runtime>(examplePolygon);
  auto e180System = e180Builder.build();
  const auto &systemDef = e180System->polygonSystem();
  const auto &detectorGeom = e180System->detectorInfo().geometry;

  auto imginfo = out_FBPImg.grid;
  int nImgWidth = imginfo.size.dimSize[0];
  int nImgHeight = imginfo.size.dimSize[1];
  int nImgDepth = imginfo.size.dimSize[2];
  float voxelSizeXY = imginfo.spacing[0];
  float voxelSizeZ = imginfo.spacing[2];

  // 构建 FBPParam (内部参数结构体)
  openpni::process::fbp::FBPParam fbpParam;
  fbpParam.nRingDiff = params.nRingDiff;
  fbpParam.dBinMin = -0.9 * examplePolygon.radius;
  fbpParam.dBinMax = 0.9 * examplePolygon.radius;
  fbpParam.nSampNumInBin = params.nSampNumInBin;
  fbpParam.nSampNumInView = params.nSampNumInView;
  fbpParam.nImgWidth = nImgWidth;
  fbpParam.nImgHeight = nImgHeight;
  fbpParam.nImgDepth = nImgDepth;
  fbpParam.voxelSizeXY = voxelSizeXY;
  fbpParam.voxelSizeZ = voxelSizeZ;
  fbpParam.deltalim = params.deltalim;
  fbpParam.klim = params.klim;
  fbpParam.wlim = params.wlim;
  fbpParam.sampling_distance_in_s = params.sampling_distance_in_s;
  fbpParam.detectorLen = params.detectorLen;

  openpni::basic::Image3DGeometry outputGeometry;
  outputGeometry.voxelSize = {voxelSizeXY, voxelSizeXY, voxelSizeZ};
  outputGeometry.voxelNum = {nImgWidth, nImgHeight, nImgDepth};
  outputGeometry.imgBegin = {-(nImgWidth * voxelSizeXY) / 2.0f, -(nImgHeight * voxelSizeXY) / 2.0f,
                             -(nImgDepth * voxelSizeZ) / 2.0f};
  if (params.rebinMethod == FBP_RebinMethod::SSRB) {
    openpni::process::fbp::FBP_SSRB(fbpParam, h_michValue, systemDef, detectorGeom, out_FBPImg.ptr, outputGeometry);
  } else if (params.rebinMethod == FBP_RebinMethod::FORE) {
    openpni::process::fbp::FBP_FORE(fbpParam, h_michValue, systemDef, detectorGeom, out_FBPImg.ptr, outputGeometry);
  }
}
void instant_FDK_CUDA(
    core::Image3DOutput<float> out_FDKImg, FDK_params params, io::U16Image const &ctImage,
    io::U16Image const &airImage) {
  using CTRawDataType = io::U16Image;
  auto imginfo = out_FDKImg.grid;
  int nImgWidth = imginfo.size.dimSize[0];
  int nImgHeight = imginfo.size.dimSize[1];
  int nImgDepth = imginfo.size.dimSize[2];
  float voxelSizeXY = imginfo.spacing[0];
  float voxelSizeZ = imginfo.spacing[2];

  openpni::basic::Image3DGeometry outputGeometry;
  outputGeometry.voxelSize = {voxelSizeXY, voxelSizeXY, voxelSizeZ};
  outputGeometry.voxelNum = {nImgWidth, nImgHeight, nImgDepth};
  outputGeometry.imgBegin = {-(nImgWidth * voxelSizeXY) / 2.0f, -(nImgHeight * voxelSizeXY) / 2.0f,
                             -(nImgDepth * voxelSizeZ) / 2.0f};
  auto d_imgOut = make_cuda_sync_ptr<float>(outputGeometry.totalVoxelNum());
  d_imgOut.allocator().memset(0, d_imgOut.span());

  // Convert uint16_t projection data to float on host before copying to device
  auto projections_float = std::vector<float>(ctImage.cspan().begin(), ctImage.cspan().end());
  auto d_projections = make_cuda_sync_ptr_from_hcopy(std::span<const float>(projections_float));

  // Convert uint16_t air data to float on host before copying to device
  auto airPixelSize = airImage.imageGeometry().voxelNum.x * airImage.imageGeometry().voxelNum.y;
  auto air_float = std::vector<float>(airImage.cspan().subspan(0, airPixelSize).begin(),
                                      airImage.cspan().subspan(0, airPixelSize).end());
  auto d_air = make_cuda_sync_ptr_from_hcopy(std::span<const float>(air_float));

  openpni::process::CBCTDataView dataView;
  std::vector<process::CBCTProjectionInfo> projectionInfos(ctImage.imageGeometry().voxelNum.z);
  for (const auto angleIndex : std::views::iota(0ull, projectionInfos.size())) {
    auto &projectionInfo = projectionInfos[angleIndex];
    const auto rotationAngle = (angleIndex * 360. / projectionInfos.size()) * M_PI / 180.0;
    const auto originDirectionU = basic::make_vec3<float>(1, 0, 0);
    const auto originDirectionV = basic::make_vec3<float>(0, 0, 1);
    const auto originPositionD = basic::make_vec3<float>(0, params.geo_SOD - params.geo_SDD, 0);
    const auto originPositionX = basic::make_vec3<float>(0, params.geo_SOD, 0);
    const auto rotation = basic::rotationMatrixZ<float>(rotationAngle);
    projectionInfo.directionU = originDirectionU * rotation;
    projectionInfo.directionV = originDirectionV * rotation;
    projectionInfo.positionD = originPositionD * rotation;
    projectionInfo.positionX = originPositionX * rotation;
  }
  const auto projectionDataPtrs = [&]() {
    std::vector<float *> result;
    const std::size_t count = ctImage.imageGeometry().voxelNum.z;
    const std::size_t step = ctImage.imageGeometry().voxelNum.x * ctImage.imageGeometry().voxelNum.y;
    for (std::size_t i = 0; i < count * step; i += step)
      result.push_back(d_projections.get() + i);
    return result;
  }();
  const auto airDataPtrs = std::vector<float *>(ctImage.imageGeometry().voxelNum.z, d_air.get());
  const auto d_projectionInfos = make_cuda_sync_ptr_from_hcopy(projectionInfos);

  dataView.pixels = basic::make_vec2<unsigned>(ctImage.imageGeometry().voxelNum.x, ctImage.imageGeometry().voxelNum.y);
  dataView.pixelSize = basic::make_vec2<float>(params.pixelSizeU, params.pixelSizeV);
  dataView.airDataPtrs =
      reinterpret_cast<const float *const *>(airDataPtrs.data()); // airDataPtrs already contains float*
  dataView.geo_angle = params.geo_angle;
  dataView.geo_offsetU = params.geo_offsetU;
  dataView.geo_offsetV = params.geo_offsetV;
  dataView.geo_SDD = params.geo_SDD;
  dataView.geo_SOD = params.geo_SOD;
  dataView.fouriorCutoffLength = params.fouriorCutoffLength;
  dataView.beamHardenParamA = params.beamHardenParamA;
  dataView.beamHardenParamB = params.beamHardenParamB;

  const basic::Image3DGeometry sliceImageGeometry = basic::make_ImageSizeByCenter(
      basic::make_vec3<float>(voxelSizeXY, voxelSizeXY, voxelSizeZ), basic::make_vec3<float>(0, 0, 0),
      basic::make_vec3<int>(nImgWidth, nImgHeight, projectionInfos.size()));
  io::F32Image sliceOneByOneImage(sliceImageGeometry);
  for (const auto sliceIndex : std::views::iota(0ull, projectionInfos.size())) {
    dataView.projectionDataPtrs = reinterpret_cast<const float *const *>(
        projectionDataPtrs.data() + sliceIndex); // projectionDataPtrs already contains float*
    dataView.projectionInfo = d_projectionInfos.get() + sliceIndex;
    dataView.projectionNum = 1;
    process::FDK_CUDA(dataView, d_imgOut.get(), outputGeometry);
  }
  process::FDK_PostProcessing(d_imgOut, outputGeometry, ctImage, params.ct_slope, params.ct_intercept,
                              params.co_offset_x, params.co_offset_y);

  // 从设备内存复制处理结果回主机
  io::F32Image outputImage(outputGeometry);
  d_imgOut.allocator().copy_from_device_to_host(outputImage.data(), d_imgOut.cspan());
  std::copy(outputImage.data(), outputImage.data() + outputGeometry.totalVoxelNum(), out_FDKImg.ptr);
}
} // namespace openpni::experimental::example