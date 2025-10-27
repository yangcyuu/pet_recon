#pragma once
#include <cuda_runtime.h>
#include <span>

#include "include/experimental/core/Mich.hpp"
namespace openpni::experimental::node::impl {
void d_fill_crystal_geoms(core::CrystalGeom *d_out_crystalGeoms, core::MichDefine mich,
                          core::CrystalGeom *d_in_crystalGeoms, std::span<core::UniformID const> d_in_crystalIDs);
void d_fill_crystal_geoms(core::CrystalGeom *d_out_crystalGeoms, core::MichDefine mich,
                          core::CrystalGeom *d_in_crystalGeoms, std::span<core::RectangleID const> d_in_crystalIDs);
void d_fill_crystal_geoms(core::CrystalGeom *d_out_crystalGeoms, core::MichDefine mich,
                          core::CrystalGeom *d_in_crystalGeoms, std::span<std::size_t const> d_in_lors);
void d_fill_crystal_geoms(std::span<core::MichStandardEvent> d_out_events, core::MichDefine mich,
                          core::CrystalGeom *d_in_crystalGeoms);
void h_fill_crystal_ids(core::MichStandardEvent *events, std::size_t const *lorIds, std::size_t count,
                        core::MichDefine mich, int16_t defualtTof = 0, float defualtValue = 1.0f);
void d_fill_crystal_ids(core::MichStandardEvent *d_events, std::size_t const *d_lorIds, std::size_t count,
                        core::MichDefine mich, int16_t defualtTof = 0, float defualtValue = 1.0f);

} // namespace openpni::experimental::node::impl