#pragma once
#include <memory>
#include <vector>

#include "../core/Image.hpp"
#include "../core/Mich.hpp"
namespace openpni::experimental::node {
class MichAttn_impl;
class MichAttn {
public:
  enum AttnFactorSource { Attn_CPU, Attn_GPU };
  enum AttnFactorFetchMode { FromPreBaked, FromInstant };

public:
  MichAttn(core::MichDefine __mich);
  MichAttn(std::unique_ptr<MichAttn_impl> impl);
  MichAttn(MichAttn &&other) noexcept;
  MichAttn &operator=(MichAttn &&other) noexcept;
  ~MichAttn();

public:
  core::MichDefine mich() const;
  void setFetchMode(AttnFactorFetchMode mode);

public:
  void setPreferredSource(AttnFactorSource source);
  void setMapSize(core::Grids<3> map);
  void bindHHUMap(float const *h_data);
  void bindHAttnMap(float const *h_data);
  std::unique_ptr<float[]> dumpAttnMich();

public:
  float const *getHAttnFactorsBatch(std::span<core::MichStandardEvent const> events);
  float const *getHAttnFactorsBatch(std::span<std::size_t const> lorIndices);
  float const *getDAttnFactorsBatch(std::span<core::MichStandardEvent const> events);
  float const *getDAttnFactorsBatch(std::span<std::size_t const> lorIndices);
  // temp
  float const *h_getAttnMap();
  float const *d_getAttnMap();
  float const *h_getAttnMich();
  float const *d_getAttnMich();
  core::Grids<3> getMapGrids();

private:
  std::unique_ptr<MichAttn_impl> m_impl;
};

} // namespace openpni::experimental::node
