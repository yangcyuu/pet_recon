#pragma once
#include "../basic/CpuInfo.hpp"
#include "../math/PathIntegral.hpp"
#include "PolygonalSystem.hpp"
namespace openpni::example::polygon {
inline void calculateScatterIntegrals(
    float *results, const float *img, const openpni::basic::Image3DGeometry &geometry, PolygonModel &model,
    const basic::Vec3<float> *scatterPoints, int scatterPointNum, float sampleRatio = 1.0f) {
  std::vector<basic::Vec2<basic::Vec3<float>>> paths;
  paths.resize(model.crystalNum() * scatterPointNum);
  process::for_each(
      model.crystalNum(),
      [&](size_t cryIndex) {
        auto cryInfo = model.crystalGeometry()[cryIndex];
        for (int scatterIndex = 0; scatterIndex < scatterPointNum; scatterIndex++) {
          paths[cryIndex * scatterPointNum + scatterIndex] = {cryInfo.position, scatterPoints[scatterIndex]};
        }
      },
      cpu_threads.halfThreads());
  core::calculatePathIntegrals(results, img, geometry, paths.data(), paths.size(), sampleRatio);
}
} // namespace openpni::example::polygon
