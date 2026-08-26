#include <algorithm>
#include <cstring>

#include "gemm/gemm.hpp"

// Mismo código que simple.cpp, compilado con -O3 (ver CMakeLists.txt).
namespace gemm::o3 {

#include "simple_impl.inc"

}  // namespace gemm::o3
