#pragma once
#include <memory>
#include <string>
#include <vector>

#include "../core/Mich.hpp"
namespace openpni::experimental::node {
class MichDeadTime_impl;
class MichDeadTime {
public:
  MichDeadTime(core::MichDefine mich);
  ~MichDeadTime();

public:
  void appendAcquisition(float const *prompMich, float const *delayMich, float scanTime, float activity);

public:
  std::vector<float> dumpCFDTTable();
  std::vector<float> dumpRTTable();
  void dumpToFile(std::string const &filename);
  void recoverFromFile(std::string const &filename);

private:
  std::unique_ptr<MichDeadTime_impl> m_impl;
};
} // namespace openpni::experimental::node