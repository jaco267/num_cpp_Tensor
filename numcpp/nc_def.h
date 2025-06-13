#pragma once
#include <cstdint>  // For uint32_t, int32_t, etc.
template <typename T = int>
using mat = std::vector<std::vector<T>>;
#define ASSERT_THROW(cond, msg) if (!(cond)) throw std::runtime_error(msg);

namespace nc{
extern const uint32_t E2MAX;
}