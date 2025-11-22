#pragma once

#include "define.h"
#include "image.h"
#include "utils.h"

struct ScatterParams {
  MichDefine define = E180();
  Grids<3> image_grid =
      Grids<3>::create_by_center_spacing_size({0.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f}, {320, 320, 400});
  int64_t inv_sss_factor = 32;
  int min_sector_difference = 4;
  int radial_module_num_s = 6;
  double tail_fitting_threshold = 0.95;
  double points_threshold = 0.00124;
  Vector3d energy_window = {350.0, 650.0, 0.15};
  Vector3d energy_eff_table = {0.01, 700.0, 0.01};
  std::string norm_factor_path;
  std::string attn_map_path;
  std::fstream::off_type attn_map_offset = 0;
  std::string mich_path;
  std::fstream::off_type mich_offset = 0;
  std::string attn_path;
  std::fstream::off_type attn_offset = 0;
  std::string delay_path;
  std::fstream::off_type delay_offset = 0;
};

class ScatterGenerator {
public:
  ScatterGenerator() = default;
  explicit ScatterGenerator(const MichDefine &define) :
      _define(define), _attn(std::make_unique<MichAttn>(define)), _norm(std::make_unique<MichNorm>(define)),
      _random(std::make_unique<MichRandom>(define)), _scatter(std::make_unique<MichScatter>(define)) {}
  explicit ScatterGenerator(const ScatterParams &params);

  static ScatterGenerator init_with_data(ScatterParams params, torch::Tensor mich);

  Texture3D generate(const torch::Tensor &emap, const std::optional<Grids<3>> &grids = std::nullopt);

private:
  MichDefine _define;
  std::unique_ptr<MichAttn> _attn;
  std::unique_ptr<MichNorm> _norm;
  std::unique_ptr<MichRandom> _random;
  std::unique_ptr<MichScatter> _scatter;
  Image3D<float> _attn_map;
  Grids<3> _image_grid;
  Texture3D _mich;
};
