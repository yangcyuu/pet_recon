#pragma once
#include <memory>

#include "../core/Mich.hpp"
namespace openpni::experimental::node {
class LORBatch_impl;
class LORBatch {
public:
  LORBatch(core::MichDefine __mich);
  ~LORBatch();

public:
  LORBatch &setSubsetNum(int num);
  LORBatch &setBinCut(int binCut);
  LORBatch &setMaxRingDiff(int maxRingDiff);
  LORBatch &setCurrentSubset(int subset);
  std::span<const std::size_t> nextHBatch();
  std::span<const std::size_t> nextDBatch();

private:
  std::unique_ptr<LORBatch_impl> m_impl;
};
} // namespace openpni::experimental::node
