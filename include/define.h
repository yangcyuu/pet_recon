#pragma once

#include <Pni-Config.hpp>
#include <experimental/core/Image.hpp>
#include <experimental/core/Mich.hpp>
#include <experimental/node/MichAttn.hpp>
#include <experimental/node/MichCrystal.hpp>
#include <experimental/node/MichNorm.hpp>
#include <experimental/node/MichRandom.hpp>
#include <experimental/node/MichScatter.hpp>

using Vector3f = openpni::experimental::core::Vector<float, 3>;
using Vector3d = openpni::experimental::core::Vector<double, 3>;
using MichDefine = openpni::experimental::core::MichDefine;
using RangeGenerator = openpni::experimental::core::RangeGenerator;
using MichCrystal = openpni::experimental::node::MichCrystal;
using RectangleID = openpni::experimental::core::RectangleID;
using MichInfoHub = openpni::experimental::core::MichInfoHub;
using MichAttn = openpni::experimental::node::MichAttn;
using MichNorm = openpni::experimental::node::MichNormalization;
using MichRandom = openpni::experimental::node::MichRandom;
using MichScatter = openpni::experimental::node::MichScatter;
template<typename T>
using Vector3 = openpni::experimental::core::Vector<T, 3>;
template<size_t D>
using Grids = openpni::experimental::core::Grids<D>;
template<typename T>
using RectangleGeom = openpni::experimental::core::RectangleGeom<T>;


MichDefine E180();

MichDefine D80();
