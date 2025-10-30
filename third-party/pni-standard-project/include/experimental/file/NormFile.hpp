#pragma once
#include <string>
#include <vector>
namespace openpni::experimental::file {
class MichNormalizationFile {
public:
  enum OpenMode { Read, Write };

public:
  MichNormalizationFile(
      OpenMode mode)
      : m_mode(mode) {}
  ~MichNormalizationFile() = default;

public:
  auto getCryCount() const { return m_cryCount; }
  auto getBlockFctA() const { return m_blockFctA; }
  auto getBlockFctT() const { return m_blockFctT; }
  auto getPlaneFct() const { return m_planeFct; }
  auto getRadialFct() const { return m_radialFct; }
  auto getInterferenceFct() const { return m_interferenceFct; }
  auto getCryFct() const { return m_cryFct; }
  void setCryCount(
      const std::vector<float> &v) {
    m_cryCount = v;
  }
  void setBlockFctA(
      const std::vector<float> &v) {
    m_blockFctA = v;
  }
  void setBlockFctT(
      const std::vector<float> &v) {
    m_blockFctT = v;
  }
  void setPlaneFct(
      const std::vector<float> &v) {
    m_planeFct = v;
  }
  void setRadialFct(
      const std::vector<float> &v) {
    m_radialFct = v;
  }
  void setInterferenceFct(
      const std::vector<float> &v) {
    m_interferenceFct = v;
  }
  void setCryFct(
      const std::vector<float> &v) {
    m_cryFct = v;
  }

public:
  void open(std::string path);

private:
  void openRead(std::string path);
  void openWrite(std::string path);

private:
  void readV1(std::stringstream &&stream);

private:
  const OpenMode m_mode;

  std::vector<float> m_cryCount;
  std::vector<float> m_blockFctA;
  std::vector<float> m_blockFctT;
  std::vector<float> m_planeFct;
  std::vector<float> m_radialFct;
  std::vector<float> m_interferenceFct;
  std::vector<float> m_cryFct;
};

} // namespace openpni::experimental::file
