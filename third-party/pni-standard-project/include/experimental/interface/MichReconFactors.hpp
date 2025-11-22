#include "../core/Mich.hpp"
namespace openpni::experimental::interface {
class MichReconFactors {
public:
  MichReconFactors() = default;
  ~MichReconFactors() = default;

public:
  enum FactorType { Addition, Multiplication };
  virtual FactorType getFactorType() const = 0;
  virtual float const *getHFactors(std::span<core::MichStandardEvent const>) const = 0;
  virtual float const *getDFactors(std::span<core::MichStandardEvent const>) const = 0;
};
} // namespace openpni::experimental::interface
