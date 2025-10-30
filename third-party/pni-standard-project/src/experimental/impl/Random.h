#pragma once
#include <cuda_runtime.h>
#include <span>

#include "include/experimental/core/Mich.hpp"
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
void dumpFactorsAsDMich(float const *factors, core::MichDefine mich, float *d_out, int minSectorDifference);
void getDRandomFactors(std::span<core::MichStandardEvent const> d_events, float const *factors, core::MichDefine mich,
                       int minSectorDifference, float *d_out);

__PNI_CUDA_MACRO__ inline float get_factor(
    core::RectangleID const &rid1, core::RectangleID const &rid2, float const *factors, core::MichDefine const &mich,
    int minSectorDifference) {
  int panel1 = rid1.idInRing / MichInfoHub(mich).getCrystalNumYInPanel();
  int panel2 = rid2.idInRing / MichInfoHub(mich).getCrystalNumYInPanel();
  if (IndexConverter(mich).isGoodPairMinSector(panel1, panel2, minSectorDifference))
    return factors[IndexConverter(mich).getFlatIdFromRectangleId(rid1)] *
           factors[IndexConverter(mich).getFlatIdFromRectangleId(rid2)];
  else
    return 0.f;
}

__PNI_CUDA_MACRO__ inline float get_factor(
    core::MichStandardEvent const &event, float const *factors, core::MichDefine const &mich, int minSectorDifference) {
  return get_factor(event.crystal1, event.crystal2, factors, mich, minSectorDifference);
}

} // namespace openpni::experimental::node::impl
