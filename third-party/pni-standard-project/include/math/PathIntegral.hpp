#pragma once
#include "../basic/Math.hpp"
#include "../process/Foreach.hpp"
#include "./Interpolation.hpp"
#include "./Sampling.hpp"

namespace openpni::core {
inline void calculatePathIntegrals(
    float *results, const float *img, const openpni::basic::Image3DGeometry &geometry,
    const openpni::basic::Vec2<openpni::basic::Vec3<float>> *points, size_t num_paths, float sampleRatio = 1.0f) {
  openpni::process::for_each(
      num_paths,
      [&](size_t i) {
        // 初始化采样器和插值器
        openpni::math::SamplingUniform<float> sampler;
        sampler.setSampleRatio(sampleRatio);
        openpni::math::InterpolationNearest3D interpolator;

        const auto &point1 = points[i].x;
        const auto &point2 = points[i].y;

        // 为采样器设置当前路径信息
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

        //  循环采样和积分
        float integral_sum = 0;
        while (!sampler.isEnd()) {
          // 获取下一个采样点
          openpni::basic::Vec3<float> sample_point = sampler.next();

          // 在采样点处进行插值，获取图像值
          float value = interpolator(sample_point, img, geometry);

          // 累加到积分和中（值 * 步长）
          integral_sum += value * sampler.getlastStepSize();
        }
        results[i] = integral_sum;
      },
      cpu_threads.halfThreads());
}
} // namespace openpni::core