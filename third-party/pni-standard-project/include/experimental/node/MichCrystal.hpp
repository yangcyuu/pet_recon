#pragma once
#include <memory>
#include <vector>

#include "../core/Mich.hpp"
namespace openpni::experimental::node {
class MichCrystal_impl;
class MichCrystal {
public:
  MichCrystal(core::MichDefine __mich);
  ~MichCrystal();

public:
  core::MichDefine mich() const;

public:
  core::CrystalGeom const *getHCrystalsBatch(std::span<std::size_t const> __lors) const;
  core::CrystalGeom const *getHCrystalsBatch(std::span<core::UniformID const> __uids) const;
  core::CrystalGeom const *getHCrystalsBatch(std::span<core::RectangleID const> __rids) const;
  void fillHCrystalsBatch(std::span<core::MichStandardEvent> __events) const;
  core::CrystalGeom const *getDCrystalsBatch(std::span<std::size_t const> __lors) const;
  core::CrystalGeom const *getDCrystalsBatch(std::span<core::UniformID const> __uids) const;
  core::CrystalGeom const *getDCrystalsBatch(std::span<core::RectangleID const> __rids) const;
  void fillDCrystalsBatch(std::span<core::MichStandardEvent> __events) const;
  std::vector<core::CrystalGeom> dumpCrystalsUniformLayout() const;
  std::vector<core::CrystalGeom> dumpCrystalsRectangleLayout() const;

private:
  std::unique_ptr<MichCrystal_impl> m_impl;
};
} // namespace openpni::experimental::node
