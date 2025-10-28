#pragma once

#include <Polygon.hpp>
#include <detector/BDM2.hpp>
#include <detector/Detectors.hpp>

#include <ATen/ops/_sobol_engine_draw.h>

#include "raw_data.h"
#include "sobol.h"
#include "texture.h"
#include "utils.h"

struct RendererParameters {
  using Model = openpni::example::polygon::PolygonModel;
  using System = openpni::example::PolygonalSystem;
  using Runtime = openpni::device::bdm2::BDM2Runtime;
  using ModelBuilder = openpni::example::polygon::PolygonModelBuilder<Runtime>;
  using DetectorUnchangable = openpni::device::DetectorUnchangable;
  using Vec3 = openpni::basic::Vec3<float>;

  /** @brief 输入 MICH 数据文件路径 */
  std::string_view mich_file;
  /** @brief 文件偏移（用于跳过前部数据） */
  std::ifstream::off_type offset = 0;
  /** @brief 晶体位置采样标准差 */
  float crystal_sigma = 0.1f;
  /** @brief 每个晶体的采样 sub-LOR 数量 */
  size_t samples_per_crystal = 16;
  /** @brief 每条 sub-LOR 的采样数量 */
  size_t samples_per_lor = 16;
  /** @brief 每个切片的迭代次数 */
  size_t iter_per_slice = 10;
  /** @brief 是否启用 Adam 优化器 */
  bool use_adam = false;
  /** @brief 是否使用 Sobol 低差采样 */
  bool use_sobol = false;
  /** @brief 学习率 */
  float learning_rate = 1e-3f;
  /** @brief 是否启用 TOF 重要性采样 */
  bool enable_importance_sampling = false;
  /** @brief TOF 权重函数标准差（暂时无用，需要改为从数据中读取） */
  float tof_sigma = 0.5f;
  /** @brief TOF 偏移中心（暂时无用，需要改为从数据中读取） */
  float tof_center_offset = 0;
  /** @brief 随机种子 */
  size_t seed = 42;
  /** @brief 单体素尺寸 */
  Vec3 voxel_size = {0.5f, 0.5f, 0.5f};
  /** @brief 重建图像像素大小 */
  Vec3 image_size = {320, 320, 400};
  /** @brief 扫描系统几何描述 */
  System polygon = openpni::example::E180();
};

class Renderer {
  using Model = openpni::example::polygon::PolygonModel;
  using System = openpni::example::PolygonalSystem;
  using Runtime = openpni::device::bdm2::BDM2Runtime;
  using ModelBuilder = openpni::example::polygon::PolygonModelBuilder<Runtime>;
  using MichDefaultIndexer = openpni::example::polygon::IndexerOfSubsetForMich;
  using DataView = openpni::basic::DataViewQTY<MichDefaultIndexer, float>;
  using CrystalGeometry = openpni::basic::CrystalGeometry;
  using Vec3 = openpni::basic::Vec3<float>;

public:
  Renderer() = default;
  template<typename DetectorType = Runtime>
  explicit Renderer(const RendererParameters &parameters) :
      _model(ModelBuilder(parameters.polygon, openpni::device::detectorUnchangable<DetectorType>()).build()),
      _mich(RawPETData<float>::from_file(parameters.mich_file,
                                         {
                                             .num_bins = _model->locator().bins().size(),
                                             .num_views = _model->locator().views().size(),
                                             .num_slices = _model->locator().slices().size(),
                                         },
                                         parameters.offset)),
      _voxel_size(parameters.voxel_size), _image_size(parameters.image_size), _crystal_sigma(parameters.crystal_sigma),
      _samples_per_crystal(parameters.samples_per_crystal), _samples_per_lor(parameters.samples_per_lor),
      _iter_per_slice(parameters.iter_per_slice), _use_adam(parameters.use_adam), _use_sobol(parameters.use_sobol),
      _enable_importance_sampling(parameters.enable_importance_sampling), _tof_sigma(parameters.tof_sigma),
      _tof_center_offset(parameters.tof_center_offset),
      _final_result(1.0f, _image_size.x, _image_size.y, _image_size.z, 1, torch::kCUDA),
      _optimizer({_final_result.tensor()}, torch::optim::AdamOptions(parameters.learning_rate)) {
    torch::manual_seed(parameters.seed);
  }

  void render(std::string_view path = {});

  void render_slice(size_t index);

  void save(const std::string_view path) const { (_final_result * _mask).save_rawdata(path); }

private:
  std::unique_ptr<Model> _model =
      ModelBuilder(openpni::example::E180(), openpni::device::detectorUnchangable<Runtime>()).build();
  RawPETData<float> _mich = {
      _model->locator().bins().size(),
      _model->locator().views().size(),
      _model->locator().slices().size(),
  };
  DataView _data_view = {
      .qtyValue = _mich.data(),
      .crystalGeometry = _model->crystalGeometry().data(),
      .indexer =
          {
              .scanner = _model->polygonSystem(),
              .detector = _model->detectorInfo().geometry,
              .subsetNum = 1,
              .subsetId = 0,
              .binCut = 0,
          },
  };

  Vec3 _voxel_size = {0.5f, 0.5f, 0.5f};
  Vec3 _image_size = {320, 320, 400};
  Vec3 _crystal_size = {_model->detectorInfo().geometry.crystalSizeU, _model->detectorInfo().geometry.crystalSizeV,
                        0.0f};
  float _crystal_sigma = 0.1f;
  size_t _samples_per_crystal = 16;
  size_t _samples_per_lor = 16;
  size_t _iter_per_slice = 10;
  bool _use_adam = false;
  bool _use_sobol = false;
  bool _enable_importance_sampling = false;
  float _tof_sigma = 0.5f;
  float _tof_center_offset = 0;

  size_t _curr_iter = 0;
  size_t _curr_slice = 0;
  bool _rendering_uniform = false;

  Texture3D _final_result = Texture3D(1.0f, _image_size.x, _image_size.y, _image_size.z, 1, torch::kCUDA);

  Texture3D _mask = Texture3D(false, _image_size.x, _image_size.y, _image_size.z, 1, torch::dtype(torch::kBool).device(torch::kCUDA));

  torch::optim::Adam _optimizer{{_final_result.tensor()}, torch::optim::AdamOptions(1e-3f)};

  SobolEngine _crystal0_sobol = SobolEngine(3, true);
  SobolEngine _crystal1_sobol = SobolEngine(3, true);

  // void render_crystal(const CrystalGeometry &start, const CrystalGeometry &end,
  //                     size_t bin_index, size_t view_index,
  //                     const Texture3D &source, DiffImage2D<float> &result);

  template<std::ranges::view T>
  void render_lor(T lor_indices, const Texture3D &source, const Texture3D &uniform_source, Texture2D &result,
                  Texture2D &uniform_result);

  torch::Tensor render_crystal(const torch::Tensor &p0, const torch::Tensor &p1, const torch::Tensor &p0u,
                               const torch::Tensor &p0v, const torch::Tensor &p1u, const torch::Tensor &p1v,
                               const torch::Tensor &crystal0_samples, const torch::Tensor &crystal1_samples,
                               const torch::Tensor &tof_samples, const Texture3D &source) const;


  torch::Tensor tof_weight(const torch::Tensor &x) const {
    if (_tof_sigma <= 0) {
      return torch::ones_like(x);
    }
    return 1 / (std::sqrt(2 * std::numbers::pi_v<float>) * _tof_sigma) *
           torch::exp(-0.5f * (x - _tof_center_offset) * (x - _tof_center_offset) / (_tof_sigma * _tof_sigma));
  }
};
