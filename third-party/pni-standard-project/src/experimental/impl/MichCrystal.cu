#include "MichCrystal.h"
#include "include/experimental/tools/Parallel.cuh"
#include "include/experimental/tools/Parallel.hpp"
namespace openpni::experimental::node::impl {
void d_fill_crystal_geoms(
    core::CrystalGeom *d_out_crystalGeoms, core::MichDefine mich, core::CrystalGeom *d_in_crystalGeoms,
    std::span<core::UniformID const> d_in_crystalIDs) {
  const auto indexConverter = core::IndexConverter::create(mich);
  tools::parallel_for_each_CUDA(
      d_in_crystalIDs.size(), [=, in_crystalIDs = d_in_crystalIDs.data()] __device__(std::size_t i) {
        d_out_crystalGeoms[i] = d_in_crystalGeoms[indexConverter.getFlatIdFromUniformID(in_crystalIDs[i])];
      });
}
void d_fill_crystal_geoms(
    core::CrystalGeom *d_out_crystalGeoms, core::MichDefine mich, core::CrystalGeom *d_in_crystalGeoms,
    std::span<core::RectangleID const> d_in_crystalIDs) {
  const auto indexConverter = core::IndexConverter::create(mich);
  tools::parallel_for_each_CUDA(
      d_in_crystalIDs.size(), [=, in_crystalIDs = d_in_crystalIDs.data()] __device__(std::size_t i) {
        d_out_crystalGeoms[i] = d_in_crystalGeoms[indexConverter.getFlatIdFromRectangleId(in_crystalIDs[i])];
      });
}
void d_fill_crystal_geoms(
    core::CrystalGeom *d_out_crystalGeoms, core::MichDefine mich, core::CrystalGeom *d_in_crystalGeoms,
    std::span<std::size_t const> d_in_lors) {
  const auto indexConverter = core::IndexConverter::create(mich);
  tools::parallel_for_each_CUDA(d_in_lors.size(), [=, in_lors = d_in_lors.data()] __device__(std::size_t i) {
    const auto rid_pair = indexConverter.getCrystalIDFromLORID(in_lors[i]);
    d_out_crystalGeoms[i * 2] = d_in_crystalGeoms[indexConverter.getFlatIdFromRectangleId(rid_pair[0])];
    d_out_crystalGeoms[i * 2 + 1] = d_in_crystalGeoms[indexConverter.getFlatIdFromRectangleId(rid_pair[1])];
  });
}
void d_fill_crystal_geoms(
    std::span<core::MichStandardEvent> d_out_events, core::MichDefine mich, core::CrystalGeom *d_in_crystalGeoms) {
  const auto indexConverter = core::IndexConverter::create(mich);
  tools::parallel_for_each_CUDA(d_out_events.size(), [=, out_events = d_out_events.data()] __device__(std::size_t i) {
    out_events[i].geo1 = d_in_crystalGeoms[indexConverter.getFlatIdFromRectangleId(out_events[i].crystal1)];
    out_events[i].geo2 = d_in_crystalGeoms[indexConverter.getFlatIdFromRectangleId(out_events[i].crystal2)];
  });
}

void h_fill_crystal_ids(
    core::MichStandardEvent *events, std::size_t const *lorIds, std::size_t count, core::MichDefine mich,
    int16_t defualtTof, float defualtValue) {
  const auto indexConverter = core::IndexConverter::create(mich);
  tools::parallel_for_each(count, [=](std::size_t i) {
    const auto rid_pair = indexConverter.getCrystalIDFromLORID(lorIds[i]);
    events[i].crystal1 = rid_pair[0];
    events[i].crystal2 = rid_pair[1];
    events[i].tof = defualtTof;
    events[i].value = defualtValue;
  });
}

void d_fill_crystal_ids(
    core::MichStandardEvent *d_events, std::size_t const *d_lorIds, std::size_t count, core::MichDefine mich,
    int16_t defualtTof, float defualtValue) {
  const auto indexConverter = core::IndexConverter::create(mich);
  tools::parallel_for_each_CUDA(count, [=, lorIds = d_lorIds, events = d_events] __device__(std::size_t i) {
    const auto rid_pair = indexConverter.getCrystalIDFromLORID(lorIds[i]);
    events[i].crystal1 = rid_pair[0];
    events[i].crystal2 = rid_pair[1];
    events[i].tof = defualtTof;
    events[i].value = defualtValue;
  });
}
} // namespace openpni::experimental::node::impl