/*! @file all2all_tanh.cc
 *  @brief "All to all" neural network layer with Tanh activation function
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */


#include <cmath>
#include <simd/memory.h>
#include <simd/arithmetic-inl.h>
#include <veles/make_unique.h>
#include "src/all2all_tanh.h"

namespace Veles {
namespace Znicz {

std::string All2AllTanh::Name() const noexcept {
  return "All2AllTanh";
}

void All2AllTanh::ApplyActivationFunction(float* data, size_t length) const {
  auto tmp = std::uniquify(mallocf(length), std::free);
  real_multiply_scalar(data, length, kScaleX, tmp.get());
  for(size_t i = 0; i < length; ++i) {
    tmp.get()[i] = std::tanh(tmp.get()[i]);
  }
  real_multiply_scalar(tmp.get(), length, kScaleY, data);
}

REGISTER_UNIT(All2AllTanh);

}  // namespace Znicz
}  // namespace Veles
