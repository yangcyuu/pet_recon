#include "../core/Geometric.hpp"
#include "../interface/SingleGenerator.hpp"
namespace openpni::experimental::node {
class BDM2R2S_impl;
class BDM2R2SArray_impl;
class BDM2R2SArray;
class BDM2R2S : public interface::SingleGenerator {
  friend class BDM2R2SArray_impl;

public:
  static core::DetectorGeom geom();
  static core::Vector<uint16_t, 2> udpPacketLengthRange();
  static char const *name() { return "BDM2"; }

public:
  BDM2R2S();
  BDM2R2S(std::unique_ptr<BDM2R2S_impl> impl);
  ~BDM2R2S();

public:
  void setChannelIndex(uint16_t channelIndex) noexcept override;
  void loadCalibration(std::string filePath) override;
  bool isCalibrationLoaded() const noexcept override;

public:
  std::span<interface::LocalSingle const> r2s_cpu(interface::PacketsInfo h_packets) const override;
  std::span<interface::LocalSingle const> r2s_cuda(interface::PacketsInfo d_packets) const override;

private:
  std::unique_ptr<BDM2R2S_impl> m_impl;
};

class BDM2R2SArray : public interface::SingleGeneratorArray {
public:
  BDM2R2SArray();
  ~BDM2R2SArray();

public:
  bool addSingleGenerator(interface::SingleGenerator *generator) noexcept override;
  void clearSingleGenerators() noexcept override;

public:
  std::span<interface::LocalSingle const> r2s_cpu(interface::PacketsInfo h_packets) const override;
  std::span<interface::LocalSingle const> r2s_cuda(interface::PacketsInfo d_packets) const override;

private:
  std::unique_ptr<BDM2R2SArray_impl> m_impl;
};

} // namespace openpni::experimental::node
