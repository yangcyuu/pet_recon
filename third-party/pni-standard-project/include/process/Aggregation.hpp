#pragma once
#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

#include "../basic/Model.hpp"
#include "../detector/Detectors.hpp"
namespace openpni::process {
struct IndexAggregator {
  struct _IndexMapper {
    std::size_t *mapper;
    std::size_t size;
    std::size_t operator[](
        std::size_t index) const {
      return mapper[index];
    }
  };

  const auto &forward() const { return forwardMapper; }
  const auto &backward() const { return backwardMapper; }

private:
  _IndexMapper forwardMapper;
  _IndexMapper backwardMapper;

private:
  std::vector<std::size_t> forwardVec;
  std::vector<std::size_t> backwardVec;

public:
  IndexAggregator(
      const std::vector<std::size_t> &countOfEachCategory) {
    const auto sum = std::accumulate(countOfEachCategory.begin(), countOfEachCategory.end(), std::size_t(0));
    forwardVec.reserve(sum);
    backwardVec.resize(countOfEachCategory.size());

    std::size_t forwardIndex = 0;
    for (std::size_t i = 0; i < countOfEachCategory.size(); ++i) {
      backwardVec[i] = forwardIndex;
      forwardIndex += countOfEachCategory[i];
    }
    for (std::size_t i = 0; i < countOfEachCategory.size(); ++i) {
      for (std::size_t j = 0; j < countOfEachCategory[i]; ++j) {
        forwardVec.push_back(i);
      }
    }

    forwardMapper.mapper = forwardVec.data();
    forwardMapper.size = forwardVec.size();
    backwardMapper.mapper = backwardVec.data();
    backwardMapper.size = backwardVec.size();
  }
};

struct _IndexAggregatorGenerator {
  IndexAggregator byBlock(
      basic::IntegratedModel model) {
    std::vector<std::size_t> countOfEachBlock;
    for (auto detector : model.detectorRuntimes()) {
      const auto geometry = detector->detectorUnchangable().geometry;
      for (const auto &blockV : std::views::iota(0, int(geometry.blockNumV)))
        for (const auto &blockU : std::views::iota(0, int(geometry.blockNumU)))
          countOfEachBlock.push_back(geometry.getCrystalNumInBlock());
    }
    return IndexAggregator(countOfEachBlock);
  }
};
inline constexpr _IndexAggregatorGenerator IndexAggregatorGenerator{};
}; // namespace openpni::process
   // namespace openpni::process
