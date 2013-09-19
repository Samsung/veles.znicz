/*! @file all2all_softmax.cc
 *  @brief "All to all" neural network layer with Softmax activation function
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
#include <algorithm>
#include <simd/memory.h>
#include <simd/mathfun.h>
#include <simd/arithmetic-inl.h>
#include <veles/make_unique.h>
#include "src/all2all_softmax.h"

namespace Veles {
namespace Znicz {

std::string All2AllSoftmax::Name() const noexcept {
  return "All2AllSoftmax";
}

void All2AllSoftmax::ApplyActivationFunction(float* data, size_t length) const {
  auto tmp_a = std::uniquify(mallocf(length), std::free);
  auto tmp_b = std::uniquify(mallocf(length), std::free);
  float max_data = *std::max_element(data, data + length);
  // tmp_a = data - Max
  for(size_t i = 0; i < length; ++i) {
    tmp_a.get()[i] = data[i] - max_data;
  }
  // tmp_b = exp(tmp_a) # using simd exp
  exp_psv(true, tmp_a.get(), length, tmp_b.get());
  // lse = ln(sum(tmp_b)) + Max # using simd sum
  float lse = log(sum(tmp_b.get(), length)) + max_data;
  // tmp_a = data - lse
  for(size_t i = 0; i < length; ++i) {
    data[i] -= lse;
  }
  // result = exp(tmp_a) # using simd exp
  exp_psv(true, tmp_a.get(), length, data);
}

REGISTER_UNIT(All2AllSoftmax);

}  // namespace Znicz
}  // namespace Veles
