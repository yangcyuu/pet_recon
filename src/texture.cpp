#include <filesystem>
#include <format>
#include <fstream>

#include "texture.h"

torch::Tensor Texture2D::eval(const torch::Tensor &uv, const TextureEvalParams &params) const {
  torch::Tensor grid = uv.unsqueeze(0).unsqueeze(2); // [1, N, 1, 2]

  torch::Tensor texture = _data.permute({2, 0, 1}).unsqueeze(0); // [1, C, H, W]

  // [1, C, N, 1]
  torch::Tensor sampled = torch::nn::functional::grid_sample(texture, grid,
                                                             torch::nn::functional::GridSampleFuncOptions()
                                                                 .mode(params.mode)
                                                                 .padding_mode(params.padding_mode)
                                                                 .align_corners(params.align_corners));

  sampled = sampled.squeeze(0).squeeze(-1).permute({1, 0}).contiguous(); // [N, C]

  return sampled;
}

void Texture2D::save_rawdata(const std::string &filename) const {
  torch::Tensor data = _data.detach().cpu();
  if (data.dtype() != torch::kFloat32) {
    data = data.to(torch::kFloat32);
  }
  data = data.contiguous();
  std::filesystem::path filepath = std::filesystem::absolute(filename);
  if (!std::filesystem::exists(filepath.parent_path())) {
    std::filesystem::create_directories(filepath.parent_path());
  }
  std::ofstream ofs(filepath, std::ios::binary);
  if (!ofs) {
    ERROR_AND_EXIT("Failed to open file for writing: {}", filepath.string());
  }
  ofs.write(static_cast<const char *>(data.data_ptr()), data.numel() * sizeof(float));
  ofs.close();
}

torch::Tensor Texture3D::eval(const torch::Tensor &uvw, const TextureEvalParams &params) const {
  torch::Tensor grid = uvw.unsqueeze(0).unsqueeze(2).unsqueeze(2); // [1, N, 1, 1, 3]

  torch::Tensor texture = _data.permute({3, 0, 1, 2}).unsqueeze(0); // [1, C, D, H, W]

  // [1, C, N, 1, 1]
  torch::Tensor sampled = torch::nn::functional::grid_sample(texture, grid,
                                                             torch::nn::functional::GridSampleFuncOptions()
                                                                 .mode(params.mode)
                                                                 .padding_mode(params.padding_mode)
                                                                 .align_corners(params.align_corners));

  sampled = sampled.squeeze(0).squeeze(-1).squeeze(-1).permute({1, 0}).contiguous(); // [N, C]

  return sampled;
}

void Texture3D::save_rawdata(const std::string &filename) const {
  torch::Tensor data = _data.detach().cpu();
  if (data.dtype() != torch::kFloat32) {
    data = data.to(torch::kFloat32);
  }
  data = data.contiguous();
  std::filesystem::path filepath = std::filesystem::absolute(filename);
  if (!std::filesystem::exists(filepath.parent_path())) {
    std::filesystem::create_directories(filepath.parent_path());
  }
  std::ofstream ofs(filepath, std::ios::binary);
  if (!ofs) {
    ERROR_AND_EXIT("Failed to open file for writing: {}", filepath.string());
  }
  ofs.write(static_cast<const char *>(data.data_ptr()), data.numel() * sizeof(float));
  ofs.close();
}
