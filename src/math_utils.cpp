#include "math_utils.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
line_in_grid(const torch::Tensor &start, const torch::Tensor &end, const Grids<3> &grids, const Vector3f &extend_size) {
  // start, end: [N, 3]
  constexpr float epsilon = 1e-6f;
  auto bounds = grids.bounding_box();
  bounds[0] = bounds[0] - extend_size;
  bounds[1] = bounds[1] + extend_size;
  torch::Tensor min_bound = torch::from_blob(bounds[0].data, {1, 3}, torch::kFloat32).to(start.options());
  torch::Tensor max_bound = torch::from_blob(bounds[1].data, {1, 3}, torch::kFloat32).to(end.options());
  torch::Tensor dir = end - start; // [N, 3]
  torch::Tensor sign_dir = 1.0f - 2.0f * torch::signbit(dir).to(torch::kFloat);
  torch::Tensor safe_dir = torch::where(torch::abs(dir) < epsilon, sign_dir * epsilon,
                                        dir); // [N, 3]
  torch::Tensor inv_dir = 1.0f / safe_dir; // [N, 3]
  torch::Tensor t0 = (min_bound - start) * inv_dir; // [N, 3]
  torch::Tensor t1 = (max_bound - start) * inv_dir; // [N, 3]
  torch::Tensor tmin = torch::minimum(t0, t1).amax(1, true); // [N, 1]
  torch::Tensor tmax = torch::maximum(t0, t1).amin(1, true); // [N, 1]
  torch::Tensor valid_mask = (tmin < tmax) & (tmax >= 0.0f) & (tmin <= 1.0f); // [N, 1]
  MARK_AS_UNUSED(tmin.clamp_(0.0f, 1.0f));
  MARK_AS_UNUSED(tmax.clamp_(0.0f, 1.0f));
  torch::Tensor clipped_start = start + dir * tmin; // [N, 3]
  torch::Tensor clipped_end = start + dir * tmax; // [N, 3]
  return std::make_tuple(clipped_start, clipped_end, valid_mask);
}
