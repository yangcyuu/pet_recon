#include "scatter_generator.h"

ScatterGenerator::ScatterGenerator(const ScatterParams &params) :
    _define(params.define),
    _attn(std::make_unique<MichAttn>(params.define)), _norm(std::make_unique<MichNorm>(params.define)),
    _random(std::make_unique<MichRandom>(params.define)), _scatter(std::make_unique<MichScatter>(params.define)),
    _image_grid(params.image_grid) {
  int64_t image_width = params.image_grid.size.dimSize[0];
  int64_t image_height = params.image_grid.size.dimSize[1];
  int64_t image_depth = params.image_grid.size.dimSize[2];
  auto generator = RangeGenerator::create(params.define);
  int64_t bin_num = generator.allBins().size();
  int64_t view_num = generator.allViews().size();
  int64_t slice_num = generator.allSlices().size();
  _attn->setFetchMode(MichAttn::FromPreBaked);
  _attn->setMapSize(_image_grid);
  _attn->setPreferredSource(MichAttn::Attn_GPU);
  if (!params.attn_map_path.empty()) {
    _attn_map =
        Image3D<float>::from_file(params.attn_map_path, image_width, image_height, image_depth, params.attn_map_offset);
    _attn->bindHAttnMap(_attn_map.data());
  }
  if (!params.norm_factor_path.empty()) {
    _norm->recoverFromFile(params.norm_factor_path);
  }
  Image3D<float> delay;
  if (!params.delay_path.empty()) {
    delay = Image3D<float>::from_file(params.delay_path, bin_num, view_num, slice_num, params.delay_offset);
    _norm->bindSelfNormMich(delay.data());
  }

  _random->setMinSectorDifference(params.min_sector_difference);
  _random->setRadialModuleNumS(params.radial_module_num_s);
  if (!params.delay_path.empty()) {
    _random->setDelayMich(delay.data());
  }

  auto sss_grid = Grids<3>::create_by_center_spacing_size(
      params.image_grid.center(), params.image_grid.spacing * static_cast<float>(params.inv_sss_factor),
      params.image_grid.size.dimSize / params.inv_sss_factor);
  _scatter->setMinSectorDifference(params.min_sector_difference);
  _scatter->setTailFittingThreshold(params.tail_fitting_threshold);
  _scatter->setScatterPointsThreshold(params.points_threshold);
  _scatter->setScatterEnergyWindow(params.energy_window);
  _scatter->setScatterEffTableEnergy(params.energy_eff_table);
  _scatter->setScatterPointGrid(sss_grid);
  _scatter->bindAttnCoff(_attn.get());
  _scatter->bindNorm(_norm.get());
  _scatter->bindRandom(_random.get());
  if (!params.mich_path.empty()) {
    auto mich_data = Image3D<float>::from_file(params.mich_path, bin_num, view_num, slice_num, params.mich_offset);
    _mich = mich_data.texture(torch::kCUDA);
    _scatter->bindDPromptMich(_mich.tensor().data_ptr<float>());
  }
}

ScatterGenerator ScatterGenerator::init_with_data(ScatterParams params, torch::Tensor mich) {
  params.mich_path.clear();
  ScatterGenerator generator(params);
  generator._mich = std::move(mich);
  generator._scatter->bindDPromptMich(generator._mich.tensor().data_ptr<float>());
  return generator;
}


Texture3D ScatterGenerator::generate(const torch::Tensor &emap, const std::optional<Grids<3>> &grids) {
  if (grids) {
    _scatter->bindDEmissionMap(*grids, emap.detach().data_ptr<float>());
  } else {
    _scatter->bindDEmissionMap(_image_grid, emap.detach().data_ptr<float>());
  }
  auto scatter = _scatter->dumpScatterMich();
  auto generator = RangeGenerator::create(_define);
  int64_t bin_num = generator.allBins().size();
  int64_t view_num = generator.allViews().size();
  int64_t slice_num = generator.allSlices().size();
  return Texture3D::cuda(scatter.get(), bin_num, view_num, slice_num, 1);
}
