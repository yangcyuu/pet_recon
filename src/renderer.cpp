#include "renderer.h"
#include "math_utils.h"

Renderer::Renderer(const RendererParams &params) :
    _image_grid(params.image_grid), _define(params.define), _psf_sigma(params.psf_sigma),
    _sub_lor_num(params.sub_lor_num), _sample_num(params.sample_num), _max_iterations(params.max_iterations),
    _batch_size(params.batch_size), _use_sobol(params.use_sobol), _linear_sampling(params.linear_sampling),
    _importance_sampling(params.importance_sampling), _compute_scatter(params.compute_scatter),
    _linear_step(params.linear_step), _tof_sigma(params.tof_sigma), _tof_center_offset(params.tof_center_offset),
    _crystal(params.define) {
  torch::manual_seed(params.seed);
  auto generator = RangeGenerator::create(params.define);
  int64_t bin_num = generator.allBins().size();
  int64_t view_num = generator.allViews().size();
  int64_t slice_num = generator.allSlices().size();
  _mich = Image3D<float>::from_file(params.mich_path, bin_num, view_num, slice_num, params.mich_offset)
              .texture(torch::kCUDA);
  if (!params.delay_path.empty()) {
    _delay = Image3D<float>::from_file(params.delay_path, bin_num, view_num, slice_num, params.delay_offset)
                 .texture(torch::kCUDA);
  }
  if (!params.attn_path.empty()) {
    _attn = Image3D<float>::from_file(params.attn_path, bin_num, view_num, slice_num, params.attn_offset)
                .texture(torch::kCUDA);
  }
  if (!params.norm_path.empty()) {
    _norm = Image3D<float>::from_file(params.norm_path, bin_num, view_num, slice_num, params.norm_offset)
                .texture(torch::kCUDA);
  }
  if (params.compute_scatter) {
    _scatter_generator = ScatterGenerator::init_with_data(params, _mich);
  }

  int64_t width = params.image_grid.size.dimSize[0];
  int64_t height = params.image_grid.size.dimSize[1];
  int64_t depth = params.image_grid.size.dimSize[2];

  _emap = Texture3D(1.0f, width, height, depth, 1, torch::kCUDA);
  _senmap = Texture3D(1.0f, width, height, depth, 1, torch::kCUDA);
  _mask = Texture3D(false, width, height, depth, 1, torch::dtype(torch::kBool).device(torch::kCUDA));
}

void Renderer::render(bool save_log) {
  const auto &generator = RangeGenerator::create(_define);
  auto lors = generator.allLORs();
  int64_t total_lors = std::ranges::distance(lors);
  int64_t bin_num = generator.allBins().size();
  int64_t view_num = generator.allViews().size();
  int64_t slice_num = generator.allSlices().size();
  const torch::Tensor &target = _mich.tensor();
  for (int64_t iter_index = 0; iter_index < _max_iterations; ++iter_index) {
    MARK_AS_UNUSED(_emap.tensor().set_requires_grad(true));
    MARK_AS_UNUSED(_senmap.tensor().set_requires_grad(true));
    _emap.tensor().mutable_grad() = torch::zeros_like(_emap.tensor());
    _senmap.tensor().mutable_grad() = torch::zeros_like(_senmap.tensor());
    int64_t batch_num = (total_lors + _batch_size - 1) / _batch_size;

    Texture3D scatter;
    if (_compute_scatter && iter_index > 0) {
      scatter = _scatter_generator.generate(_emap);
      if (save_log) {
        scatter.save_rawdata("log/scatter.image3d");
      }
    }
    for (int64_t batch_index = 0; batch_index < batch_num; ++batch_index) {
      int64_t begin_index = batch_index * _batch_size;
      int64_t end_index = std::min<int64_t>(begin_index + _batch_size, total_lors);
      auto lor_batch = lors | std::views::drop(begin_index) | std::views::take(end_index - begin_index);

      Texture3D result(0.0f, bin_num, view_num, slice_num, 1, torch::kCUDA);
      Texture3D uniform_result(0.0f, bin_num, view_num, slice_num, 1, torch::kCUDA);
      std::cout << std::format("\r  Processing batch {}/{}...\t\t", batch_index + 1, batch_num);
      if (!render_lors(lor_batch, _emap, _senmap, result, uniform_result)) {
        continue;
      }
      if (!_attn.empty()) {
        MARK_AS_UNUSED(result.tensor().mul_(_attn));
        MARK_AS_UNUSED(uniform_result.tensor().mul_(_attn));
      }
      if (!_norm.empty()) {
        MARK_AS_UNUSED(result.tensor().mul_(_norm));
        MARK_AS_UNUSED(uniform_result.tensor().mul_(_norm));
      }
      if (!_delay.empty()) {
        MARK_AS_UNUSED(result.tensor().add_(_delay));
        MARK_AS_UNUSED(uniform_result.tensor().add_(_delay));
      }
      if (_compute_scatter && iter_index > 0) {
        MARK_AS_UNUSED(result.tensor().add_(scatter));
        MARK_AS_UNUSED(uniform_result.tensor().add_(scatter));
      }
      torch::Tensor likelihood = (target * (result.tensor() + 1e-8).log() - result).sum();
      likelihood.backward();

      torch::Tensor uni_result = uniform_result.tensor().sum();
      uni_result.backward();

      torch::NoGradGuard no_grad;
      std::cout << std::format("likelihood sum = {:.6f}", likelihood.item<float>()) << std::flush;
    }
    torch::Tensor dldi = _emap.tensor().grad();
    torch::Tensor drdi = _senmap.tensor().grad();
    CHECK_TENSOR_NAN(dldi, "dL/di contains NaN values");
    CHECK_TENSOR_NAN(drdi, "dR/di contains NaN values");

    torch::NoGradGuard no_grad;
    torch::Tensor g = torch::where(drdi > 1e-6f, dldi / drdi, torch::zeros_like(drdi));
    torch::Tensor update = _emap.tensor() * g;
    _mask.tensor() |= update != 0.0f;
    MARK_AS_UNUSED(_emap.tensor().add_(update).clamp_min_(0.0f));
    std::cout << std::format("\n  Iteration {} completed.\n", iter_index + 1);
    std::cout << std::format("  dl/di min = {:.6f}, max = {:.6f}, mean = {:.6f}\n", dldi.min().item<float>(),
                             dldi.max().item<float>(), dldi.mean().item<float>());
    std::cout << std::format("  dr/di min = {:.6f}, max = {:.6f}, mean = {:.6f}\n", drdi.min().item<float>(),
                             drdi.max().item<float>(), drdi.mean().item<float>());
    std::cout << std::format("  update min = {:.6f}, max = {:.6f}, mean = {:.6f}\n", update.min().item<float>(),
                             update.max().item<float>(), update.mean().item<float>());
    std::cout << std::format("  result min = {:.6f}, max = {:.6f}, mean = {:.6f}\n", _emap.tensor().min().item<float>(),
                             _emap.tensor().max().item<float>(), _emap.tensor().mean().item<float>());
    if (save_log) {
      save(std::format("log/iter_{:03}_result.image3d", iter_index + 1));
    }
  }
}

void Renderer::save(const std::string &path) { Texture3D(_emap.tensor() * _mask).save_rawdata(path); }


torch::Tensor Renderer::tof_weight(const torch::Tensor &x) const {
  if (_tof_sigma <= 0) {
    return torch::ones_like(x);
  }
  return 1.0f / (std::sqrt(2.0f * std::numbers::pi_v<float>) * _tof_sigma) *
         torch::exp(-0.5f * (x - _tof_center_offset) * (x - _tof_center_offset) / (_tof_sigma * _tof_sigma));
}

torch::Tensor Renderer::tof_pdf(const torch::Tensor &x) const {
  if (_tof_sigma <= 0) {
    return torch::ones_like(x);
  }
  return 1.0f / (std::sqrt(2.0f * std::numbers::pi_v<float>) * _tof_sigma) *
         torch::exp(-0.5f * (x - _tof_center_offset) * (x - _tof_center_offset) / (_tof_sigma * _tof_sigma));
}

torch::Tensor Renderer::render_lor(const torch::Tensor &p0, const torch::Tensor &p1, const torch::Tensor &p0u,
                                   const torch::Tensor &p0v, const torch::Tensor &p0n, const torch::Tensor &p1u,
                                   const torch::Tensor &p1v, const torch::Tensor &p1n, const torch::Tensor &p0_samples,
                                   const torch::Tensor &p1_samples, const torch::Tensor &linear_offset,
                                   torch::Tensor tof_samples, const Texture3D &source) {
  int64_t num_lors = p0.size(0);
  torch::Device device = torch::kCUDA;

  torch::Tensor psf_sigma = torch::tensor(_psf_sigma, device);
  torch::Tensor tof_sigma = torch::tensor(_tof_sigma, device);
  torch::Tensor tof_offset = torch::tensor(_tof_center_offset, device);
  auto box_size = _image_grid.boxLength();
  torch::Tensor image_size = torch::tensor({box_size[0], box_size[1], box_size[2]}, device);
  torch::Tensor image_origin =
      torch::tensor({_image_grid.origin[0], _image_grid.origin[1], _image_grid.origin[2]}, device);
  // [N, N_sub_lor, 3]
  torch::Tensor p0_offsets = gaussian_sample(p0_samples, psf_sigma, 0.0f);
  torch::Tensor p1_offsets = gaussian_sample(p1_samples, psf_sigma, 0.0f);

  torch::Tensor p0_offsets_u = p0_offsets.select(-1, 0).unsqueeze(-1); // [N, N_sub_lor, 1]
  torch::Tensor p0_offsets_v = p0_offsets.select(-1, 1).unsqueeze(-1);
  torch::Tensor p0_offsets_n = p0_offsets.select(-1, 2).unsqueeze(-1);
  torch::Tensor p1_offsets_u = p1_offsets.select(-1, 0).unsqueeze(-1); // [N, N_sub_lor, 1]
  torch::Tensor p1_offsets_v = p1_offsets.select(-1, 1).unsqueeze(-1);
  torch::Tensor p1_offsets_n = p1_offsets.select(-1, 2).unsqueeze(-1);
  // [N, N_sub_lor, 3]
  torch::Tensor p0d = p0.unsqueeze(1) + p0_offsets_u * p0u.unsqueeze(1) + p0_offsets_v * p0v.unsqueeze(1) +
                      p0_offsets_n * p0n.unsqueeze(1);
  torch::Tensor p1d = p1.unsqueeze(1) + p1_offsets_u * p1u.unsqueeze(1) + p1_offsets_v * p1v.unsqueeze(1) +
                      p1_offsets_n * p1n.unsqueeze(1);
  torch::Tensor sub_lor_mask;
  if (!_importance_sampling || _tof_sigma <= 0) {
    std::tie(p0d, p1d, sub_lor_mask) = line_in_grid(p0d.view({-1, 3}), p1d.view({-1, 3}), _image_grid);
    p0d = p0d.view({num_lors, -1, 3});
    p1d = p1d.view({num_lors, -1, 3});
    sub_lor_mask = sub_lor_mask.view({num_lors, -1});
  } else {
    sub_lor_mask = torch::ones({num_lors, _sub_lor_num}, torch::dtype(torch::kBool).device(device));
  }


  torch::Tensor length = torch::norm(p1d - p0d, 2, -1).clamp_min_(1e-6f); // [N, N_sub_lor]
  torch::Tensor x; // [N, N_sub_lor, N_sample]
  torch::Tensor sample_mask; // [N, N_sub_lor, N_sample]
  int64_t max_sample_num;
  if (_linear_sampling) {
    // [N, N_sub_lor]
    torch::Tensor actual_sample_num = torch::ceil(length / _linear_step).to(torch::kInt64); // [N, N_sub_lor]
    max_sample_num = actual_sample_num.max().item<int64_t>();
    torch::Tensor sample_indices = torch::arange(0, max_sample_num, device).view({1, 1, -1});
    sample_mask = sample_indices < actual_sample_num.unsqueeze(-1); // [N, N_sub_lor, N_sample]
    tof_samples = (sample_indices + linear_offset) / actual_sample_num.unsqueeze(-1); // [N, N_sub_lor, N_sample]
  } else {
    max_sample_num = _sample_num;
  }

  if (_importance_sampling && _tof_sigma > 0) {
    x = gaussian_sample(tof_samples, tof_sigma, tof_offset);
  } else {
    x = (tof_samples - 0.5f) * length.unsqueeze(-1); // [N, N_sub_lor, N_sample]
  }

  torch::Tensor t = (0.5f + x / length.unsqueeze(-1)).clamp_(0.0f, 1.0f); // [N, N_sub_lor, N_sample]

  torch::Tensor p0d_exp = p0d.unsqueeze(2); // [N, N_sub_lor, 1, 3]
  torch::Tensor p1d_exp = p1d.unsqueeze(2); // [N, N_sub_lor, 1, 3]
  torch::Tensor pos = p0d_exp * (1.0f - t.unsqueeze(-1)) + p1d_exp * t.unsqueeze(-1); // [N, N_sub_lor, N_sample, 3]

  torch::Tensor coord = (pos - image_origin) / image_size; // [N, N_sub_lor, N_sample, 3]
  coord = coord * 2.0f - 1.0f; // to [-1, 1]

  torch::Tensor vals = source.eval(coord.view({-1, 3}), {.align_corners = false}); // [N * N_sub_lor * N_sample, 1]

  torch::Tensor dist = torch::norm(pos - p0d_exp, 2, -1) - length.unsqueeze(-1) / 2.0f; // [N, N_sub_lor, N_sample]
  torch::Tensor tof_weights = tof_weight(dist); // [N, N_sub_lor, N_sample]
  torch::Tensor pdfs;
  if (_importance_sampling && _tof_sigma > 0) {
    pdfs = tof_pdf(dist);
  } else {
    pdfs = 1.0f / length.unsqueeze(-1);
  }

  torch::Tensor fs =
      vals.view({num_lors, -1, max_sample_num}) * tof_weights / pdfs.clamp_min_(1e-6f); // [N, N_sub_lor, N_sample]

  torch::Tensor sum; // [N]
  torch::Tensor num;
  if (_linear_sampling) {
    sum = torch::where(sample_mask & sub_lor_mask.unsqueeze(-1), fs, torch::zeros_like(fs)).sum({1, 2});
    num = (sample_mask & sub_lor_mask.unsqueeze(-1)).to(torch::kInt64).sum({1, 2});
  } else {
    sum = torch::where(sub_lor_mask.unsqueeze(-1), fs, torch::zeros_like(fs)).sum({1, 2});
    num = sub_lor_mask.to(torch::kInt64).sum(1) * max_sample_num;
  }
  return torch::where(num > 0, sum / num, torch::zeros_like(sum));
}
