#pragma once
#include "../process/EM.hpp"
#include "PolygonalSystem.hpp"

namespace openpni::example {
inline void forwardProjection(
    float *out_fwdMich, float *in_ImgData, int bincut, const example::polygon::PolygonModel &model) {
  basic::Image3DGeometry imgGeo{{0.5f, 0.5f, 0.5f}, {-80, -80, -100}, {320, 320, 400}};
  Image3DSpan<const float> imgSpan{imgGeo, in_ImgData};
  openpni::basic::DataViewQTY<openpni::example::polygon::IndexerOfSubsetForMich, float> dataView;
  dataView.qtyValue = nullptr;
  dataView.indexer.scanner = model.polygonSystem();
  dataView.indexer.detector = model.detectorInfo().geometry;
  dataView.indexer.subsetId = 0;
  dataView.indexer.subsetNum = 1;
  dataView.indexer.binCut = bincut;

  process::EMSum(imgSpan, imgSpan.geometry.roi(), out_fwdMich, dataView, math::ProjectionMethodUniform(),
                 basic::CpuMultiThread::callWithAllThreads());
}
inline void backwardProjection(
    float *out_ImgData, float *in_fwdMich, int bincut, const example::polygon::PolygonModel &model) {
  basic::Image3DGeometry imgGeo{{0.5f, 0.5f, 0.5f}, {-80, -80, -100}, {320, 320, 400}};
  Image3DSpan<float> imgSpan{imgGeo, out_ImgData};
  openpni::basic::DataViewQTY<openpni::example::polygon::IndexerOfSubsetForMich, float> dataView;
  dataView.qtyValue = in_fwdMich;
  dataView.indexer.scanner = model.polygonSystem();
  dataView.indexer.detector = model.detectorInfo().geometry;
  dataView.indexer.subsetId = 0;
  dataView.indexer.subsetNum = 1;
  dataView.indexer.binCut = bincut;

  process::EMDistribute(in_fwdMich, imgSpan.ptr, imgSpan.geometry, imgSpan.geometry.roi(), dataView,
                        math::ProjectionMethodUniform(), basic::CpuMultiThread::callWithAllThreads());
}
} // namespace openpni::example
