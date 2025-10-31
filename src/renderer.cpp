#include "renderer.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include <format>
#include <numbers>
#include <ranges>
#include "hash.h"

void Renderer::render(const std::string_view path) {
  std::ranges::for_each(std::views::iota(0uz, _iter_per_slice), [&](size_t iter) {
    _curr_iter = iter;
    const auto &generator = _mich_range_generator;
    auto lors = generator.allLORs();
    Texture3D target = _mich.texture().tensor().to(torch::kCUDA); // [1, D, H, W]

    Texture3D &source = _final_result;
    source.set_requires_grad(true);
    source.zero_grad();

    Texture3D &uniform_source = _uniform_result;
    uniform_source.set_requires_grad(true);
    uniform_source.zero_grad();

    Texture3D result(generator.allBins().size(), generator.allViews().size(), generator.allSlices().size(), 1,
                     torch::kCUDA);

    Texture3D uniform_result(generator.allBins().size(), generator.allViews().size(), generator.allSlices().size(), 1,
                             torch::kCUDA);

    size_t batch_num = (std::ranges::distance(lors) + _batch_size - 1) / _batch_size;
    for (size_t batch_index = 0; batch_index < batch_num; ++batch_index) {
      size_t begin_index = batch_index * _batch_size;
      size_t end_index = std::min<size_t>(begin_index + _batch_size, std::ranges::distance(lors));
      auto lors_batch = lors | std::views::drop(begin_index) | std::views::take(end_index - begin_index);
      std::cout << std::format("\r  Processing batch {}/{}...", batch_index + 1, batch_num);
      render_lor(lors_batch, source, uniform_source, result, uniform_result);
    }

    Texture3D likelihood = target * result.clamp_min(1e-8f).log() - result;
    torch::Tensor mean_likelihood = likelihood.tensor().mean();
    mean_likelihood.backward();
    Texture3D dldi = source.grad();

    c10::cuda::CUDACachingAllocator::emptyCache();

    uniform_result.tensor().mean().backward();
    Texture3D drdi = uniform_source.grad();

    torch::NoGradGuard no_grad;
    Texture3D g = torch::where(drdi.tensor() > 1e-6f, (dldi / drdi).tensor(), torch::zeros_like(drdi.tensor()));
    // Texture3D update =
    //     torch::where(source.tensor() != 0.0f, (source * g).tensor(), source.tensor().mean() * g.tensor());
    Texture3D update = source * g;

    _mask = torch::where(update.tensor() != 0.0f, torch::ones_like(_mask.tensor()), _mask.tensor());
    MARK_AS_UNUSED(source.add_(update).clamp_min_(0.0f));
    std::cout << std::format("\n  Iteration {} completed.\n", iter + 1);
    std::cout << std::format("  dl/di min = {:.6f}, max = {:.6f}, mean = {:.6f}\n", dldi.tensor().min().item<float>(),
                             dldi.tensor().max().item<float>(), dldi.tensor().mean().item<float>());
    std::cout << std::format("  dr/di min = {:.6f}, max = {:.6f}, mean = {:.6f}\n", drdi.tensor().min().item<float>(),
                             drdi.tensor().max().item<float>(), drdi.tensor().mean().item<float>());
    std::cout << std::format("  update min = {:.6f}, max = {:.6f}, mean = {:.6f}\n",
                             update.tensor().min().item<float>(), update.tensor().max().item<float>(),
                             update.tensor().mean().item<float>());
    std::cout << std::format("  likelihood mean = {:.6f}, result min = {:.6f}, max = {:.6f}, mean = {:.6f}\n",
                             mean_likelihood.item<float>(), source.tensor().min().item<float>(),
                             source.tensor().max().item<float>(), source.tensor().mean().item<float>());

    save(std::format("log/iter_{:03}_result.image3d", iter + 1));
  });

  if (!path.empty()) {
    save(path);
  }
}


template<std::ranges::view T>
void Renderer::render_lor(const T &lor_indices, const Texture3D &source, const Texture3D &uniform_source,
                          Texture3D &result, Texture3D &uniform_result) {
  torch::Device device = torch::kCUDA;
  size_t num_lors = std::ranges::distance(lor_indices);
  static std::vector<int64_t> bin_indices;
  static std::vector<int64_t> view_indices;
  static std::vector<int64_t> slice_indices;
  static std::vector<float> p0_data;
  static std::vector<float> p1_data;
  static std::vector<float> p0u_data;
  static std::vector<float> p0v_data;
  static std::vector<float> p1u_data;
  static std::vector<float> p1v_data;
  bin_indices.clear();
  view_indices.clear();
  slice_indices.clear();
  p0_data.clear();
  p1_data.clear();
  p0u_data.clear();
  p0v_data.clear();
  p1u_data.clear();
  p1v_data.clear();

  bin_indices.reserve(num_lors);
  view_indices.reserve(num_lors);
  slice_indices.reserve(num_lors);
  p0_data.reserve(num_lors * 3);
  p1_data.reserve(num_lors * 3);
  p0u_data.reserve(num_lors * 3);
  p0v_data.reserve(num_lors * 3);
  p1u_data.reserve(num_lors * 3);
  p1v_data.reserve(num_lors * 3);

  std::vector<size_t> lors;
  lors.reserve(num_lors);
  for (size_t i = 0; i < num_lors; ++i) {
    lors.push_back(lor_indices[i]);
  }

  auto crystal_geometries = _mich_crystal.getHCrystalsBatch(lors);

  const auto &generator = _mich_range_generator;
  size_t clipped_lor_count = 0;
  for (size_t lor_index = 0; lor_index < num_lors; ++lor_index) {
    size_t bin_index = lors[lor_index] % generator.allBins().size();
    size_t view_index = lors[lor_index] / generator.allBins().size() % generator.allViews().size();
    size_t slice_index =
        lors[lor_index] / (generator.allBins().size() * generator.allViews().size()) % generator.allSlices().size();
    auto position0 = crystal_geometries[2 * lor_index].O;
    auto position1 = crystal_geometries[2 * lor_index + 1].O;
    const auto &direction_u0 = crystal_geometries[2 * lor_index].U;
    const auto &direction_v0 = crystal_geometries[2 * lor_index].V;
    const auto &direction_u1 = crystal_geometries[2 * lor_index + 1].U;
    const auto &direction_v1 = crystal_geometries[2 * lor_index + 1].V;
    const auto lor_check = lor_in_image(position0, position1, _voxel_size, _image_size);
    if (!lor_check.has_value()) {
      clipped_lor_count++;
      continue;
    }
    const auto &[tmin, tmax] = lor_check.value();
    // Clip the LOR
    const auto clip_position0 = position0 + (position1 - position0) * tmin;
    const auto clip_position1 = position0 + (position1 - position0) * tmax;
    position0 = clip_position0;
    position1 = clip_position1;
    bin_indices.push_back(static_cast<int64_t>(bin_index));
    view_indices.push_back(static_cast<int64_t>(view_index));
    slice_indices.push_back(static_cast<int64_t>(slice_index));
    // p0_data.insert(p0_data.end(), {position0[0], position0[1], position0[2]});
    // p1_data.insert(p1_data.end(), {position1[0], position1[1], position1[2]});
    // p0u_data.insert(p0u_data.end(), {direction_u0[0], direction_u0[1], direction_u0[2]});
    // p0v_data.insert(p0v_data.end(), {direction_v0[0], direction_v0[1], direction_v0[2]});
    // p1u_data.insert(p1u_data.end(), {direction_u1[0], direction_u1[1], direction_u1[2]});
    // p1v_data.insert(p1v_data.end(), {direction_v1[0], direction_v1[1], direction_v1[2]});
    p0_data.push_back(position0[0]);
    p0_data.push_back(position0[1]);
    p0_data.push_back(position0[2]);
    p1_data.push_back(position1[0]);
    p1_data.push_back(position1[1]);
    p1_data.push_back(position1[2]);
    p0u_data.push_back(direction_u0[0]);
    p0u_data.push_back(direction_u0[1]);
    p0u_data.push_back(direction_u0[2]);
    p0v_data.push_back(direction_v0[0]);
    p0v_data.push_back(direction_v0[1]);
    p0v_data.push_back(direction_v0[2]);
    p1u_data.push_back(direction_u1[0]);
    p1u_data.push_back(direction_u1[1]);
    p1u_data.push_back(direction_u1[2]);
    p1v_data.push_back(direction_v1[0]);
    p1v_data.push_back(direction_v1[1]);
    p1v_data.push_back(direction_v1[2]);
  }

  std::cout << std::format(" {}/{} LORs clipped.\t\t\t", clipped_lor_count, num_lors) << std::flush;
  num_lors -= clipped_lor_count;

  if (num_lors == 0) {
    return;
  }

  torch::Tensor p0 = torch::from_blob(p0_data.data(), {static_cast<int64_t>(num_lors), 3}, torch::kFloat32).to(device);
  torch::Tensor p1 = torch::from_blob(p1_data.data(), {static_cast<int64_t>(num_lors), 3}, torch::kFloat32).to(device);
  torch::Tensor p0u =
      torch::from_blob(p0u_data.data(), {static_cast<int64_t>(num_lors), 3}, torch::kFloat32).to(device);
  torch::Tensor p0v =
      torch::from_blob(p0v_data.data(), {static_cast<int64_t>(num_lors), 3}, torch::kFloat32).to(device);
  torch::Tensor p1u =
      torch::from_blob(p1u_data.data(), {static_cast<int64_t>(num_lors), 3}, torch::kFloat32).to(device);
  torch::Tensor p1v =
      torch::from_blob(p1v_data.data(), {static_cast<int64_t>(num_lors), 3}, torch::kFloat32).to(device);


  torch::Tensor bin_idx =
      torch::from_blob(bin_indices.data(), {static_cast<int64_t>(num_lors)}, torch::kInt64).to(device);
  torch::Tensor view_idx =
      torch::from_blob(view_indices.data(), {static_cast<int64_t>(num_lors)}, torch::kInt64).to(device);
  torch::Tensor slice_idx =
      torch::from_blob(slice_indices.data(), {static_cast<int64_t>(num_lors)}, torch::kInt64).to(device);

  // [N, N_crystal, 3]
  torch::Tensor crystal0_samples;
  torch::Tensor crystal1_samples;

  if (_use_sobol) {
    crystal0_samples = _crystal0_sobol.draw(num_lors * _samples_per_crystal)
                           .reshape({static_cast<int64_t>(num_lors), static_cast<int64_t>(_samples_per_crystal), 3})
                           .to(device);
    crystal1_samples = _crystal1_sobol.draw(num_lors * _samples_per_crystal)
                           .reshape({static_cast<int64_t>(num_lors), static_cast<int64_t>(_samples_per_crystal), 3})
                           .to(device);
  } else {
    crystal0_samples =
        torch::rand({static_cast<int64_t>(num_lors), static_cast<int64_t>(_samples_per_crystal), 3}, device);
    crystal1_samples =
        torch::rand({static_cast<int64_t>(num_lors), static_cast<int64_t>(_samples_per_crystal), 3}, device);
  }

  //[N, N_crystal, N_lor]
  torch::Tensor tof_samples;

  if (!_linear_sampling) {
    tof_samples = torch::rand({static_cast<int64_t>(num_lors), static_cast<int64_t>(_samples_per_crystal),
                               static_cast<int64_t>(_samples_per_lor)},
                              device);
  }

  torch::Tensor values =
      render_crystal(p0, p1, p0u, p0v, p1u, p1v, crystal0_samples, crystal1_samples, tof_samples, source); // [N]
  result.assign(slice_idx, view_idx, bin_idx, torch::zeros_like(bin_idx), values);

  torch::Tensor uniform_values = render_crystal(p0, p1, p0u, p0v, p1u, p1v, crystal0_samples, crystal1_samples,
                                                tof_samples, uniform_source); // [N]
  uniform_result.assign(slice_idx, view_idx, bin_idx, torch::zeros_like(bin_idx), uniform_values);
}


torch::Tensor Renderer::render_crystal(const torch::Tensor &p0, const torch::Tensor &p1, const torch::Tensor &p0u,
                                       const torch::Tensor &p0v, const torch::Tensor &p1u, const torch::Tensor &p1v,
                                       const torch::Tensor &crystal0_samples, const torch::Tensor &crystal1_samples,
                                       torch::Tensor tof_samples, const Texture3D &source) const {
  torch::Device device = torch::kCUDA;
  int64_t num_lors = p0.size(0);

  torch::Tensor crystal_sigma = torch::tensor(_crystal_sigma, device);
  torch::Tensor tof_sigma_t = torch::tensor(_tof_sigma, device);
  torch::Tensor tof_offset_t = torch::tensor(_tof_center_offset, device);

  torch::Tensor voxel_size = torch::tensor({_voxel_size[0], _voxel_size[1], _voxel_size[2]}, device);
  torch::Tensor image_size = torch::tensor({_image_size[0], _image_size[1], _image_size[2]}, device);

  // [N, N_crystal, 3]
  torch::Tensor crystal0_offsets = gaussian_sample(crystal0_samples, crystal_sigma);
  torch::Tensor crystal1_offsets = gaussian_sample(crystal1_samples, crystal_sigma);

  torch::Tensor crystal0_offsets_u = crystal0_offsets.select(-1, 0).unsqueeze(-1); // [N, N_crystal, 1]
  torch::Tensor crystal0_offsets_v = crystal0_offsets.select(-1, 1).unsqueeze(-1);
  torch::Tensor crystal1_offsets_u = crystal1_offsets.select(-1, 0).unsqueeze(-1); // [N, N_crystal, 1]
  torch::Tensor crystal1_offsets_v = crystal1_offsets.select(-1, 1).unsqueeze(-1);

  // [N, N_crystal, 3]
  torch::Tensor p0d = p0.unsqueeze(1) + crystal0_offsets_u * p0u.unsqueeze(1) + crystal0_offsets_v * p0v.unsqueeze(1);
  torch::Tensor p1d = p1.unsqueeze(1) + crystal1_offsets_u * p1u.unsqueeze(1) + crystal1_offsets_v * p1v.unsqueeze(1);


  // [N, N_crystal, 1]
  torch::Tensor length = torch::norm(p1d - p0d, 2, 2, true);

  torch::Tensor x; // [N, N_crystal, N_lor]
  torch::Tensor mask; // [N, N_crystal, N_lor]
  torch::Tensor actual_samples_num; // [N, N_crystal, 1]
  size_t actual_samples_per_lor;
  if (_linear_sampling) {
    actual_samples_num = torch::ceil(length / _linear_step); // [N, N_crystal, 1]
    size_t max_samples_num = actual_samples_num.max().item<int64_t>();
    torch::Tensor sample_indices =
        torch::arange(0, static_cast<int64_t>(max_samples_num), device)
            .reshape({1, 1, -1})
            .expand({num_lors, static_cast<int64_t>(_samples_per_crystal), static_cast<int64_t>(max_samples_num)});
    mask = sample_indices < actual_samples_num; // [N, N_crystal, N_lor]
    actual_samples_per_lor = max_samples_num;
    tof_samples = (sample_indices + 0.5f) / actual_samples_num; // [N, N_crystal, N_lor]
  } else {
    actual_samples_per_lor = _samples_per_lor;
  }

  if (_enable_importance_sampling && _tof_sigma > 0) {
    x = gaussian_sample(tof_samples, tof_sigma_t, tof_offset_t);
  } else {
    x = (tof_samples - 0.5f) * length;
  }

  // [N, N_crystal, N_lor, 1]
  torch::Tensor t = (0.5f + x / length).unsqueeze(-1);
  t = torch::clamp(t, 0.0f, 1.0f);

  // [N, N_crystal, N_lor, 3]
  torch::Tensor p0d_exp = p0d.unsqueeze(2); // [N, N_crystal, 1, 3]
  torch::Tensor p1d_exp = p1d.unsqueeze(2); // [N, N_crystal, 1, 3]
  torch::Tensor pos = p0d_exp * (1.0f - t) + p1d_exp * t;

  // [N, N_crystal, N_lor]
  torch::Tensor dist = torch::norm(pos - p0d_exp, 2, 3) - length / 2.0f;
  torch::Tensor tof_w = tof_weight(dist);

  // [N, N_crystal, N_lor, 3]
  torch::Tensor coord = (pos / voxel_size.unsqueeze(0).unsqueeze(0).unsqueeze(0) + 0.5f) /
                            image_size.unsqueeze(0).unsqueeze(0).unsqueeze(0) +
                        0.5f;

  int64_t total_samples = num_lors * _samples_per_crystal * actual_samples_per_lor;
  torch::Tensor coord_flat = coord.reshape({total_samples, 3});


  // [total_samples, C]
  torch::Tensor f_flat = source.eval_coords(coord_flat);

  // [N, N_crystal, N_lor, C]
  torch::Tensor f = f_flat.reshape(
      {num_lors, static_cast<int64_t>(_samples_per_crystal), static_cast<int64_t>(actual_samples_per_lor), -1});

  torch::Tensor pdf;
  if (_enable_importance_sampling && _tof_sigma > 0) {
    pdf = 1.0f / (std::sqrt(2.0f * std::numbers::pi_v<float>) * _tof_sigma) *
          torch::exp(-0.5f * (x - _tof_center_offset) * (x - _tof_center_offset) /
                     (_tof_sigma * _tof_sigma)); // [N, N_crystal, N_lor]
  } else {
    pdf = 1.0f / length.squeeze(-1); // [N, N_crystal]
    pdf = pdf.unsqueeze(2); // [N, N_crystal, 1]
  }

  // [N, N_crystal, N_lor]
  torch::Tensor tof_w_exp = tof_w.unsqueeze(-1);
  torch::Tensor pdf_exp = pdf.unsqueeze(-1);
  torch::Tensor weighted = (f * tof_w_exp / (pdf_exp + 1e-6f)).squeeze(-1); // [N, N_crystal, N_lor]

  torch::Tensor lor_mean;

  if (_linear_sampling) {
    lor_mean =
        torch::where(actual_samples_num.squeeze(-1) > 0, (weighted * mask).sum(2) / actual_samples_num.squeeze(-1),
                     torch::zeros_like(actual_samples_num.squeeze(-1)));
  } else {
    lor_mean = weighted.mean(2); // [N, N_crystal]
  }


  // [N, C]
  torch::Tensor crystal_mean = lor_mean.mean(1);

  // [N]
  torch::Tensor result = crystal_mean.squeeze(-1);

  return result;
}
