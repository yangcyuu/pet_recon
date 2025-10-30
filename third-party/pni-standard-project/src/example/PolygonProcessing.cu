#include "include/example/ConvolutionKernel.hpp"
#include "include/example/PolygonProcessing.cuh"
#include "include/process/EM.cuh"
namespace openpni::example::polygon {
process::LocalSinglesOfEachChannel RtoS_DData(
    PolygonModel &model, process::RawDataView d_rawDataView) {
  auto detectors = model.detectorRuntimes();
  return process::rtos(d_rawDataView, detectors.data());
}
process::LocalSinglesOfEachChannel RtoS_HData(
    PolygonModel &model, process::RawDataView h_rawDataView) {
  auto dp_channel = make_cuda_sync_ptr_from_hcopy<uint16_t>(std::span(h_rawDataView.channel, h_rawDataView.count));
  auto dp_data = make_cuda_sync_ptr_from_hcopy<uint8_t>(
      std::span(h_rawDataView.data,
                h_rawDataView.length[h_rawDataView.count - 1] + h_rawDataView.offset[h_rawDataView.count - 1]));
  auto dp_length = make_cuda_sync_ptr_from_hcopy<uint16_t>(std::span(h_rawDataView.length, h_rawDataView.count));
  auto dp_offset = make_cuda_sync_ptr_from_hcopy<uint64_t>(std::span(h_rawDataView.offset, h_rawDataView.count));
  process::RawDataView d_data;
  d_data.channel = dp_channel.get();
  d_data.channelNum = h_rawDataView.channelNum;
  d_data.clock_ms = h_rawDataView.clock_ms;
  d_data.count = h_rawDataView.count;
  d_data.data = dp_data.get();
  d_data.duration_ms = h_rawDataView.duration_ms;
  d_data.length = dp_length.get();
  d_data.offset = dp_offset.get();
  return RtoS_DData(model, d_data);
}

process::CoincidenceResult StoC_DData(
    PolygonModel &model, const process::LocalSinglesOfEachChannel &d_globalSingles,
    process::CoincidenceProtocol protocol) {
  return process::stoc(d_globalSingles, model.detectorRuntimes().data(), protocol);
}

std::vector<unsigned> rearrange_countmap(
    const std::vector<unsigned> &countmap, PolygonModel &model) {
  const auto layerSize = model.crystalNum();
  auto copy = countmap;
  for (const auto layer : std::views::iota(0u, 4u))
    for (const auto [rect, uni, u, v] : model.locator().allCrystalUniformAndRectangleAndUV())
      copy[rect + layer * layerSize] = countmap[uni + layer * layerSize];
  return copy;
}

void listmode_to_mich(
    std::vector<float> &mich, PolygonModel &model, std::span<const basic::Listmode_t> listmodes) {
  for (const auto listmode : listmodes) {
    const auto [cry1, cry2, dt] = listmode;
    mich[example::polygon::calLORIDFromCrystalUniformID(model.polygonSystem(), model.detectorInfo().geometry, cry1,
                                                        cry2)]++;
  }
}

void fill_senmap_cuda(
    PolygonModel &model, io::F32Image &senmap, int subsetId, int subsetNum, float hfwhm) {
  if (!senmap.elements())
    return;
  if (subsetNum == 0)
    return;
  if (subsetId < 0 || subsetId >= subsetNum)
    return;

  auto d_senmap = make_cuda_sync_ptr_from_hcopy<float>(senmap.elements());
  d_senmap.allocator().memset(0, d_senmap.span());
  auto d_crystalGeometry = make_cuda_sync_ptr_from_hcopy<basic::CrystalGeometry>(model.crystalGeometry());
  auto d_convolutionKernel =
      make_cuda_sync_ptr_from_hcopy(example::gaussianKernel<9>(senmap.imageGeometry().voxelSize / hfwhm));
  openpni::basic::DataViewQTY<openpni::example::polygon::IndexerOfSubsetForMich, float> dataView;
  dataView.qtyValue = nullptr;
  dataView.indexer.scanner = model.polygonSystem();
  dataView.indexer.detector = model.detectorInfo().geometry;
  dataView.indexer.subsetId = subsetId;
  dataView.indexer.subsetNum = subsetNum;
  dataView.indexer.binCut = 0;
  dataView.crystalGeometry = d_crystalGeometry;
  process::calSenmap_CUDA(dataView, senmap.imageGeometry(), d_senmap.data(), d_convolutionKernel.get(),
                          math::ProjectionMethodUniform());
  d_senmap.allocator().copy_from_device_to_host(senmap.data(), d_senmap.cspan());
}
} // namespace openpni::example::polygon