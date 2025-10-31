#pragma once

#include <PnI-Config.hpp>
#include <experimental/node/MichCrystal.hpp>

#include "cgmath.h"
#include "define.h"
#include "raw_data.h"
#include "sobol.h"
#include "texture.h"
#include "utils.h"

struct RendererParameters {
  using Vector3 = openpni::experimental::core::Vector<float, 3>;
  using MichDefine = openpni::experimental::core::MichDefine;

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
  /** @brief 每次处理的 LOR 数量 */
  size_t batch_size = 32768;
  /** @brief 是否使用 Sobol 低差采样 */
  bool use_sobol = false;
  /** @brief 是否使用线性采样 */
  bool linear_sampling = false;
  /** @brief 线性采样步长 */
  float linear_step = 1.0f;
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
  Vector3 voxel_size = {0.5f, 0.5f, 0.5f};
  /** @brief 重建图像像素大小 */
  Vector3 image_size = {320, 320, 400};
  /** @brief 扫描系统几何描述 */
  MichDefine define = E180();
};

class Renderer {
  using Vector3 = openpni::experimental::core::Vector<float, 3>;
  using MichDefine = openpni::experimental::core::MichDefine;
  using RangeGenerator = openpni::experimental::core::RangeGenerator;
  using MichCrystal = openpni::experimental::node::MichCrystal;
  using RectangleID = openpni::experimental::core::RectangleID;
  using RectangleGeom = openpni::experimental::core::RectangleGeom<float>;

public:
  Renderer() = default;
  explicit Renderer(const RendererParameters &parameters) :
      _mich_range_generator(RangeGenerator::create(parameters.define)),
      _mich(RawPETData<float>::from_file(parameters.mich_file,
                                         {
                                             .num_bins = _mich_range_generator.allBins().size(),
                                             .num_views = _mich_range_generator.allViews().size(),
                                             .num_slices = _mich_range_generator.allSlices().size(),
                                         },
                                         parameters.offset)),
      _mich_crystal(MichCrystal(parameters.define)), _voxel_size(parameters.voxel_size),
      _image_size(parameters.image_size), _crystal_sigma(parameters.crystal_sigma),
      _samples_per_crystal(parameters.samples_per_crystal), _samples_per_lor(parameters.samples_per_lor),
      _iter_per_slice(parameters.iter_per_slice), _batch_size(parameters.batch_size), _use_sobol(parameters.use_sobol),
      _linear_sampling(parameters.linear_sampling), _linear_step(parameters.linear_step),
      _enable_importance_sampling(parameters.enable_importance_sampling), _tof_sigma(parameters.tof_sigma),
      _tof_center_offset(parameters.tof_center_offset) {
    torch::manual_seed(parameters.seed);
  }

  void render(std::string_view path = {});


  void save(const std::string_view path) const { (_final_result * _mask).save_rawdata(path); }

private:
  RangeGenerator _mich_range_generator = RangeGenerator::create(E180());
  RawPETData<float> _mich = {
      _mich_range_generator.allBins().size(),
      _mich_range_generator.allViews().size(),
      _mich_range_generator.allSlices().size(),
  };

  MichCrystal _mich_crystal = MichCrystal(E180());

  Vector3 _voxel_size = {0.5f, 0.5f, 0.5f};
  Vector3 _image_size = {320, 320, 400};

  float _crystal_sigma = 0.1f;
  size_t _samples_per_crystal = 16;
  size_t _samples_per_lor = 16;
  size_t _iter_per_slice = 10;
  size_t _batch_size = 32768;
  bool _use_sobol = false;
  bool _linear_sampling = false;
  float _linear_step = 1.0f;
  bool _enable_importance_sampling = false;
  float _tof_sigma = 0.5f;
  float _tof_center_offset = 0;

  size_t _curr_iter = 0;
  size_t _curr_slice = 0;
  bool _rendering_uniform = false;

  Texture3D _final_result = Texture3D(1.0f, _image_size[0], _image_size[1], _image_size[2], 1, torch::kCUDA);
  Texture3D _uniform_result = Texture3D(1.0f, _image_size[0], _image_size[1], _image_size[2], 1, torch::kCUDA);

  Texture3D _mask = Texture3D(false, _image_size[0], _image_size[1], _image_size[2], 1,
                              torch::dtype(torch::kBool).device(torch::kCUDA));


  SobolEngine _crystal0_sobol = SobolEngine(3, true);
  SobolEngine _crystal1_sobol = SobolEngine(3, true);

  // void render_crystal(const CrystalGeometry &start, const CrystalGeometry &end,
  //                     size_t bin_index, size_t view_index,
  //                     const Texture3D &source, DiffImage2D<float> &result);

  template<std::ranges::view T>
  void render_lor(const T &lor_indices, const Texture3D &source, const Texture3D &uniform_source, Texture3D &result,
                  Texture3D &uniform_result);

  torch::Tensor render_crystal(const torch::Tensor &p0, const torch::Tensor &p1, const torch::Tensor &p0u,
                               const torch::Tensor &p0v, const torch::Tensor &p1u, const torch::Tensor &p1v,
                               const torch::Tensor &crystal0_samples, const torch::Tensor &crystal1_samples,
                               torch::Tensor tof_samples, const Texture3D &source) const;


  torch::Tensor tof_weight(const torch::Tensor &x) const {
    if (_tof_sigma <= 0) {
      return torch::ones_like(x);
    }
    return 1 / (std::sqrt(2 * std::numbers::pi_v<float>) * _tof_sigma) *
           torch::exp(-0.5f * (x - _tof_center_offset) * (x - _tof_center_offset) / (_tof_sigma * _tof_sigma));
  }
};
