#pragma once
#ifndef PNI_STANDARD_BASIC_EXCEPTIONS_INCLUDE
#define PNI_STANDARD_BASIC_EXCEPTIONS_INCLUDE
#include <exception>
#define DEFINE_EXCEPTION(exceptionName)                                                                                \
                                                                                                                       \
  class exceptionName : public std::runtime_error {                                                                    \
  public:                                                                                                              \
    exceptionName()                                                                                                    \
        : std::runtime_error("openpni::exceptions::" #exceptionName)                                                   \
        , m_message("openpni::exceptions::" #exceptionName) {}                                                         \
    exceptionName(                                                                                                     \
        std::string __message)                                                                                         \
        : std::runtime_error("openpni::exceptions::" #exceptionName " " + __message)                                   \
        , m_message("openpni::exceptions::" #exceptionName " " + __message) {}                                         \
    virtual ~exceptionName() {}                                                                                        \
    virtual const char *what() const throw() override { return m_message.c_str(); }                                    \
                                                                                                                       \
  private:                                                                                                             \
    std::string m_message;                                                                                             \
  };

namespace openpni::exceptions {
DEFINE_EXCEPTION(
    file_cannot_access)
DEFINE_EXCEPTION(
    file_path_empty)
DEFINE_EXCEPTION(
    file_format_incorrect)
DEFINE_EXCEPTION(
    file_unknown_version)
DEFINE_EXCEPTION(
    file_type_mismatch)
DEFINE_EXCEPTION(
    file_flags_invalid)
DEFINE_EXCEPTION(
    resource_unavailable)
DEFINE_EXCEPTION(
    invalid_environment)

DEFINE_EXCEPTION(cuda_error_invalid_value);
DEFINE_EXCEPTION(cuda_error_invalid_memcpy_direction);
DEFINE_EXCEPTION(cuda_error_not_support);
DEFINE_EXCEPTION(cuda_error_memory_allocation);
DEFINE_EXCEPTION(cuda_error_unknown);
DEFINE_EXCEPTION(cuda_error_device_unavailable);
DEFINE_EXCEPTION(cuda_error_stream_incompatible);

DEFINE_EXCEPTION(json_parse_error);

DEFINE_EXCEPTION(algorithm_unexpected_condition);

} // namespace openpni::exceptions
#undef DEFINE_EXCEPTION
#endif // !PNI_STANDARD_BASIC_EXCEPTIONS_INCLUDE
