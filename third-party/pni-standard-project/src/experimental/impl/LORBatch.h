#pragma once
#include <cuda_runtime.h>
#include <span>

#include "include/experimental/core/Mich.hpp"
namespace openpni::experimental::node::impl {
void d_fillLORBatch(std::size_t begin, std::size_t end, int binCut, int subsetNum, int currentSubset,
                    std::span<core::Vector<int64_t, 2> const> d_ringPair, core::MichDefine mich, std::size_t *d_out);
} // namespace openpni::experimental::node::impl
