#pragma once
#include "../basic/Math.hpp"
#include "../process/Foreach.cuh"
#include "./Interpolation.hpp"
#include "./Sampling.hpp"

namespace openpni::core {
inline void calculatePathIntegrals_CUDA(
    float *results, const float *img, const openpni::basic::Image3DGeometry &geometry,
    const openpni::basic::Vec2<openpni::basic::Vec3<float>> *points, size_t num_paths, float sampleRatio = 1.0f,
    cudaStream_t stream = 0) {
  openpni::process::for_each_CUDA(
      num_paths,
      [=] __device__(size_t i) {
        // 采样器和插值器必须为每个线程独立创建
        openpni::math::SamplingUniform<float> sampler;
        sampler.setSampleRatio(sampleRatio);
        openpni::math::InterpolationNearest3D interpolator;

        const auto &point1 = points[i].x;
        const auto &point2 = points[i].y;

        // 构造路径事件
        openpni::basic::CrystalGeometry crystal_geom1{.position = point1};
        openpni::basic::CrystalGeometry crystal_geom2{.position = point2};
        openpni::basic::CrystalInfo crystal_info1{.geometry = &crystal_geom1};
        openpni::basic::CrystalInfo crystal_info2{.geometry = &crystal_geom2};
        openpni::basic::Event<float> path_event{
            .crystal1 = crystal_info1, .crystal2 = crystal_info2, .eventIndex = (uint32_t)i};

        if (!sampler.setInfo(path_event, &geometry)) {
          results[i] = float(0);
          return;
        }

        float integral_sum = 0;
        while (!sampler.isEnd()) {
          openpni::basic::Vec3<float> sample_point = sampler.next();
          float value = interpolator(sample_point, img, geometry);
          integral_sum += value * sampler.getlastStepSize();
        }
        results[i] = integral_sum;
      },
      stream);
}
} // namespace openpni::core