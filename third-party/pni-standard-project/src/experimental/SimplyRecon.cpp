#include "include/experimental/example/SimplyRecon.hpp"

#include <fstream>

#include "impl/Share.hpp"
#include "impl/Test.h"
#include "impl/WrappedConv3D.h"
#include "include/experimental/algorithms/CalGeometry.hpp"
#include "include/experimental/algorithms/PathIntegral.hpp"
#include "include/experimental/example/EasyParallel.hpp"
#include "include/experimental/node/GaussConv3D.hpp"
#include "include/experimental/node/LORBatch.hpp"
#include "include/experimental/node/MichCrystal.hpp"
#include "include/experimental/node/Senmaps.hpp"
#include "include/experimental/tools/Loop.hpp"
#include "include/io/Decoding.hpp"
#include "include/io/IO.hpp"
#include "include/misc/CycledBuffer.hpp"
#include "include/misc/ListmodeBuffer.hpp"
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

  {
    // if (michAttn) {
    //   auto ptr = michAttn->dumpAttnMich();
    //   std::ofstream attnFile("AttnMich_dump.bin", std::ios::binary);
    //   if (attnFile.is_open()) {
    //     attnFile.write(reinterpret_cast<const char *>(ptr.get()), MichInfoHub(mich).getLORNum() * sizeof(float));
    //     attnFile.close();
    //     std::cout << "Attn Mich dumped to AttnMich_dump.bin" << std::endl;
    //   }
    // }
  }

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
        d_parralel_mul(d_tempValues.get(), d_tempMichValue.get(), d_tempValues.get(), lors.size());
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

        d_parralel_fill(d_tempMichValue.get(), 1.0f, currentBatchSize);
        node::impl::d_fill_standard_events_ids_from_listmodeTOF(
            d_tempEvents, context.Listmode_data.cspan().subspan(batchStart, currentBatchSize), params.TOF_division,
            mich);
        michCrystal.fillDCrystalsBatch(d_tempEvents.span());
        node::impl::d_simple_path_integral_batch_TOF(core::Image3DInput<float>{d_out_OSEMImg.grid, d_tmp1.get()},
                                                     d_tempEvents.cspan(), params.sample_rate, d_tempValues.get());
        node::impl::d_apply_correction_factor(d_tempValues, d_tempEvents.span(currentBatchSize), michNorm, michRand,
                                              michScat, michAttn, d_factors_add, d_factors_mul);
        node::impl::d_osem_fix_integral_value(d_tempValues.span(currentBatchSize));
        d_parralel_mul(d_tempValues.get(), d_tempMichValue.get(), d_tempValues.get(), currentBatchSize);
        node::impl::d_fill_standard_events_values(d_tempEvents.get(), d_tempValues.get(), currentBatchSize);
        node::impl::d_simple_path_reverse_integral_batch_TOF(
            core::Image3DOutput<float>{d_out_OSEMImg.grid, d_tmp2.get()}, d_tempEvents.cspan(currentBatchSize),
            params.sample_rate);
      }

      convolver.deconvD(core::Image3DIO<float>{d_out_OSEMImg.grid, d_tmp2.get(), d_tmp2.get()});
      senmap.updateDImage(d_tmp2.get(), d_out_OSEMImg.ptr, subsetId, d_out_OSEMImg.grid);
    }
all_end:
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

        d_parralel_fill(d_tempMichValue.get(), 1.0f, currentBatchSize);

        node::impl::d_fill_standard_events_ids_from_listmode(
            d_tempEvents, context.Listmode_data.cspan().subspan(batchStart, currentBatchSize), mich);

        michCrystal.fillDCrystalsBatch(d_tempEvents.span());

        node::impl::d_simple_path_integral_batch(core::Image3DInput<float>{d_out_OSEMImg.grid, d_tmp1.get()},
                                                 d_tempEvents.cspan(), params.sample_rate, d_tempValues.get());
        node::impl::d_apply_correction_factor(d_tempValues, d_tempEvents.span(currentBatchSize), michNorm, michRand,
                                              michScat, michAttn, d_factors_add, d_factors_mul);
        node::impl::d_osem_fix_integral_value(d_tempValues.span(currentBatchSize));
        d_parralel_mul(d_tempValues.get(), d_tempMichValue.get(), d_tempValues.get(), currentBatchSize);
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
    d_parralel_fill(d_out_OSEMImg.get(), 1.0f, grid.totalSize());
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
  for (const auto &randomListmodeFile : randomListmodeFiles)
    if (randomListmodeFile.size()) {
      result.michRand = std::make_unique<node::MichRandom>(mich);
      result.michRand->setMinSectorDifference(minSector);
      result.michRand->setRadialModuleNumS(radialModeulNumS);
      openpni::io::ListmodeFileInput delayListmodeFile;
      delayListmodeFile.open(randomListmodeFile);
      auto selectedDelayListmodeSegments =
          openpni::io::selectSegments(delayListmodeFile, listmodeFileTimeBegin_ms, listmodeFileTimeEnd_ms);
      auto totalDelayEvents =
          std::accumulate(selectedDelayListmodeSegments.begin(), selectedDelayListmodeSegments.end(), 0ull,
                          [](auto a, auto b) { return a + b.dataIndexEnd - b.dataIndexBegin; });
      PNI_DEBUG("Delay Listmode file opened, reading segments...\n");

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
          michRand->setRandomRatio(double(__count) / double(totalEvents));
        for (auto scatterIter = 0; scatterIter <= params.scatterSimulations; scatterIter++) {
          d_parralel_fill(d_out_OSEMImg.get(), 1.0f, grid.totalSize());
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
        if (michRand)
          michRand->setRandomRatio(double(__count) / double(totalEvents));

        for (auto scatterIter = 0; scatterIter <= params.scatterSimulations; scatterIter++) {
          d_parralel_fill(d_out_OSEMImg.get(), 1.0f, grid.totalSize());
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
  int gpuNum = 0;
  cudaGetDeviceCount(&gpuNum);

  PNI_DEBUG(std::format("System GPU num = {}\n", gpuNum));
  if (gpuNum <= 0)
    throw std::runtime_error("No CUDA device found.");

  common::MultiCycledBuffer<MultiGPUData> cycleBuffer(gpuNum);
  std::vector<std::thread> threadsGPU;
  std::mutex h_addMutex;
  for (int i = 0; i < gpuNum; ++i) {
    threadsGPU.emplace_back([&, i]() {
      cudaSetDevice(i);

      auto norm = params.doNorm ? std::make_unique<node::MichNormalization>(mich) : nullptr;
      if (norm) {
        norm->recoverFromFile(params.normFactorsFile);
        if (params.doSelfNorm)
          norm->bindSelfNormMich(params.selfNormMich);
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
        scat->bindAttnCoff(attn.get());
        scat->bindNorm(norm.get());
        scat->bindRandom(rand.get());
      }

      node::GaussianConv3D conv3D;
      conv3D.setHWHM(params.gauss_hwhm_mm);

      OSEM_context_listmode context;
      context.michCrystal = std::make_unique<node::MichCrystal>(mich);
      context.senmap = std::make_unique<node::MichSenmap>(conv3D, mich);
      context.senmap->setSubsetNum(params.subsetNum);
      context.senmap->setMode(node::MichSenmap::Mode_listmode);
      if (norm)
        context.senmap->bindNormalization(norm.get());
      if (attn)
        context.senmap->bindAttenuation(attn.get());
      context.convolver = &conv3D;

      PNI_DEBUG(std::format("Thread for GPU {} started.\n", i));
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

        if (scat)
          scat->bindDListmode(context.Listmode_data.span(context.events_count));
        scat->setTOFParams(params.tofSSS_timeBinWidth_ps, params.tofSSS_timeBinStart_ps, params.tofSSS_timeBinEnd_ps,
                           params.tofSSS_systemTimeRes_ns);
        if (rand) {
          rand->setRandomRatio(double(context.events_count) / double(totalEvents));
          PNI_DEBUG(std::format("GPU {} set random ratio to {}, events_count {} total {}\n", i,
                                double(context.events_count) / double(totalEvents), context.events_count, totalEvents));
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
        _params.TOF_division = params.TOF_division;
        _params.binCutRatio = params.binCutRatio;
        _params.scatterSimulations = params.scatterSimulations;
        _params.listmodeFileTimeBegin_ms = params.timeBegin_ms;
        _params.listmodeFileTimeEnd_ms = params.timeEnd_ms;

        if (!scat)
          params.scatterSimulations = 0;

        for (auto scatterIter = 0; scatterIter <= params.scatterSimulations; scatterIter++) {
          auto osem_grid = grid;
          if (scat && scatterIter < params.scatterSimulations)
            osem_grid = bigGrid;
          d_parralel_fill(d_out_OSEMImg.get(), 1.0f, grid.totalSize());
          osem_impl_listmodeTOF_CUDA(core::Image3DOutput<float>{osem_grid, d_out_OSEMImg}, _params, t_norm, t_rand,
                                     t_scat, t_attn, mich, context);
          d_out_OSEMImg.allocator().copy_from_device_to_device(d_out_OSEMImgPair.data(), d_out_OSEMImg.cspan());
          if (scat && scatterIter < params.scatterSimulations)
            scat->bindDEmissionMap(osem_grid, d_out_OSEMImgPair);
        }

        PNI_DEBUG(std::format("Chunk processing complete.\n"));
        auto h_img = make_vector_from_cuda_sync_ptr(d_out_OSEMImg, d_out_OSEMImg.cspan());
        std::lock_guard lock(h_addMutex);
        tools::parallel_for_each(d_out_OSEMImg.elements(), [&](size_t idx) { out_OSEMImg.ptr[idx] += h_img[idx]; });
        PNI_DEBUG(std::format("GPU {} image max value: {}.\n", i,
                              *std::max_element(h_img.data(), h_img.data() + grid.totalSize())));
      }))
        ;
      PNI_DEBUG(std::format("Thread for GPU {} ending.\n", i));
    });
  }
  misc::ListmodeBuffer listmodeBuffer;
  listmodeBuffer
      .setBufferSize(find_value(totalEvents, _GB(params.size_GB) / sizeof(openpni::basic::Listmode_t), gpuNum))
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
  PNI_DEBUG("OSEM_listModeTOF_CUDA done.\n");
  cycleBuffer.stop();

  for (auto &t : threadsGPU)
    t.join();
}
} // namespace openpni::experimental::example