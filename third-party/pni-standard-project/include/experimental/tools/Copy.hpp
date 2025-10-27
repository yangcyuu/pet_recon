#pragma once
namespace openpni::experimental::tools {
class DisableCopy {
public:
  DisableCopy() = default;
  DisableCopy(const DisableCopy &) = delete;
  DisableCopy &operator=(const DisableCopy &) = delete;
  DisableCopy(DisableCopy &&) = default;
  DisableCopy &operator=(DisableCopy &&) = default;
  ~DisableCopy() = default;
};
} // namespace openpni::experimental::tools
