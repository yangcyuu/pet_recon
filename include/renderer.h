#pragma once

#include "crystal_buffer.h"
#include "define.h"
#include "math_utils.h"
#include "scatter_generator.h"
#include "sobol.h"
#include "texture.h"

struct RendererParams : ScatterParams {
  std::string norm_path;
  std::fstream::off_type norm_offset = 0;
  float psf_sigma = 0.0f;
  int64_t sub_lor_num = 16;
  int64_t sample_num = 16;
  int64_t max_iterations = 10;
  int64_t batch_size = 65536;
  bool use_sobol = false;
  bool linear_sampling = false;
  bool importance_sampling = false;
  bool compute_scatter = false;
  float linear_step = 1.0f;
  float tof_sigma = 0.0f;
  float tof_center_offset = 0.0f;
  size_t seed = 42;
};

class Renderer {

public:
  Renderer() = default;

  explicit Renderer(const RendererParams &params);

  void render(bool save_log = false);

  void save(const std::string &path);

private:
  Grids<3> _image_grid;
  MichDefine _define;
  float _psf_sigma = 0.0f;
  int64_t _sub_lor_num = 16;
  int64_t _sample_num = 16;
  int64_t _max_iterations = 10;
  int64_t _batch_size = 65536;
  bool _use_sobol = false;
  bool _linear_sampling = false;
  bool _importance_sampling = false;
  bool _compute_scatter = false;
  float _linear_step = 1.0f;
  float _tof_sigma = 0.0f;
  float _tof_center_offset = 0.0f;

  MichCrystal _crystal = MichCrystal(E180());
  ScatterGenerator _scatter_generator;
  CrystalBuffer<float> _crystal_buffer;

  SobolEngine _p0_sobol = SobolEngine(3, true);
  SobolEngine _p1_sobol = SobolEngine(3, true);
  SobolEngine _linear_sobol = SobolEngine(1, true);

  Texture3D _mich;
  Texture3D _norm;
  Texture3D _attn;
  Texture3D _delay;

  Texture3D _emap;
  Texture3D _senmap;
  Texture3D _mask;

  torch::Tensor tof_weight(const torch::Tensor &x) const;
  torch::Tensor tof_pdf(const torch::Tensor &x) const;

  torch::Tensor render_lor(const torch::Tensor &p0, const torch::Tensor &p1, const torch::Tensor &p0u,
                           const torch::Tensor &p0v, const torch::Tensor &p0n, const torch::Tensor &p1u,
                           const torch::Tensor &p1v, const torch::Tensor &p1n, const torch::Tensor &p0_samples,
                           const torch::Tensor &p1_samples, const torch::Tensor &linear_offset,
                           torch::Tensor tof_samples, const Texture3D &source);

  template<std::ranges::view T>
  bool render_lors(const T &lor_indices, const Texture3D &source, const Texture3D &uniform_source, Texture3D &result,
                   Texture3D &uniform_result);
};

template<std::ranges::view T>
bool Renderer::render_lors(const T &lor_indices, const Texture3D &source, const Texture3D &uniform_source,
                           Texture3D &result, Texture3D &uniform_result) {
  torch::Device device = torch::kCUDA;
  int64_t num_lors = std::ranges::distance(lor_indices);

  auto generator = RangeGenerator::create(_define);
  int64_t bin_num = generator.allBins().size();
  int64_t view_num = generator.allViews().size();
  int64_t slice_num = generator.allSlices().size();

  std::vector<size_t> lor_idx;
  lor_idx.reserve(num_lors);
  for (size_t i = 0; i < num_lors; ++i) {
    lor_idx.push_back(lor_indices[i]);
  }

  auto crystals = _crystal.getHCrystalsBatch(lor_idx);
  _crystal_buffer.crystals({crystals, 2 * lor_idx.size()});

  auto p0_data = _crystal_buffer.p0();
  auto p0u_data = _crystal_buffer.u0();
  auto p0v_data = _crystal_buffer.v0();
  auto p1_data = _crystal_buffer.p1();
  auto p1u_data = _crystal_buffer.u1();
  auto p1v_data = _crystal_buffer.v1();
  torch::Tensor lors = torch::from_blob(lor_idx.data(), {num_lors}, torch::kInt64).to(device);
  torch::Tensor p0 = torch::from_blob(p0_data.data(), {num_lors, 3}, torch::kFloat32).to(device);
  torch::Tensor p0u = torch::from_blob(p0u_data.data(), {num_lors, 3}, torch::kFloat32).to(device);
  torch::Tensor p0v = torch::from_blob(p0v_data.data(), {num_lors, 3}, torch::kFloat32).to(device);
  torch::Tensor p1 = torch::from_blob(p1_data.data(), {num_lors, 3}, torch::kFloat32).to(device);
  torch::Tensor p1u = torch::from_blob(p1u_data.data(), {num_lors, 3}, torch::kFloat32).to(device);
  torch::Tensor p1v = torch::from_blob(p1v_data.data(), {num_lors, 3}, torch::kFloat32).to(device);

  float crystal_size = (_define.detector.crystalSizeU + _define.detector.crystalSizeV) * 0.5f;
  Vector3f extend_size = Vector3f::create(crystal_size) * 0.5f + Vector3f::create(_psf_sigma) * 3.0f;
  // [N, 1]
  torch::Tensor lor_mask;
  std::tie(std::ignore, std::ignore, lor_mask) = line_in_grid(p0, p1, _image_grid, extend_size);
  lors = lors.masked_select(lor_mask.squeeze(-1));
  if (lors.size(0) == 0) {
    return false;
  }
  num_lors = lors.size(0);
  p0 = p0.masked_select(lor_mask).view({-1, 3});
  p0u = p0u.masked_select(lor_mask).view({-1, 3});
  p0v = p0v.masked_select(lor_mask).view({-1, 3});
  p1 = p1.masked_select(lor_mask).view({-1, 3});
  p1u = p1u.masked_select(lor_mask).view({-1, 3});
  p1v = p1v.masked_select(lor_mask).view({-1, 3});
  torch::Tensor bin_idx = lors % bin_num;
  torch::Tensor view_idx = lors.div(bin_num, "trunc") % view_num;
  torch::Tensor slice_idx = lors.div(bin_num * view_num, "trunc") % slice_num;

  torch::Tensor p0n = 2.0f * torch::cross(p0u, p0v, 1); // [N, 3]
  torch::Tensor p1n = 2.0f * torch::cross(p1u, p1v, 1); // [N, 3]

  // [N, N_sub_lor, 3]
  torch::Tensor p0_samples;
  torch::Tensor p1_samples;
  if (_use_sobol) {
    p0_samples = _p0_sobol.draw(num_lors * _sub_lor_num).view({num_lors, _sub_lor_num, -1}).to(device);
    p1_samples = _p1_sobol.draw(num_lors * _sub_lor_num).view({num_lors, _sub_lor_num, -1}).to(device);
  } else {
    p0_samples = torch::rand({num_lors, _sub_lor_num, 3}, device);
    p1_samples = torch::rand({num_lors, _sub_lor_num, 3}, device);
  }

  // [N, N_sub_lor, 1]
  torch::Tensor linear_offset;
  // [N, N_sub_lor, N_sample]
  torch::Tensor tof_samples;
  if (!_linear_sampling) {
    tof_samples = torch::rand({num_lors, _sub_lor_num, _sample_num}, device);
  } else {
    linear_offset = _linear_sobol.draw(_sub_lor_num).view({1, _sub_lor_num, -1}).to(device);
  }

  torch::Tensor values = render_lor(p0, p1, p0u, p0v, p0n, p1u, p1v, p1n, p0_samples, p1_samples, linear_offset,
                                    tof_samples, source); // [N]
  result.assign(slice_idx, view_idx, bin_idx, torch::zeros_like(bin_idx), values);

  torch::Tensor uniform_values = render_lor(p0, p1, p0u, p0v, p0n, p1u, p1v, p1n, p0_samples, p1_samples, linear_offset,
                                            tof_samples, uniform_source); // [N]
  uniform_result.assign(slice_idx, view_idx, bin_idx, torch::zeros_like(bin_idx), uniform_values);
  return true;
}
