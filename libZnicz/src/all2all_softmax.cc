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

namespace veles {
namespace znicz {

std::string All2AllSoftmax::Name() const noexcept {
  return "All2AllSoftmax";
}

void All2AllSoftmax::ApplyActivationFunction(float* data, size_t length) const {
  auto a_minus_max = std::uniquify(mallocf(length), std::free);
  float max_data = *std::max_element(data, data + length);
  // a_minus_max = data - Max
  for(size_t i = 0; i < length; ++i) {
    a_minus_max.get()[i] = data[i] - max_data;
  }
  // data = exp(a_minus_max) # using simd exp
  exp_psv(true, a_minus_max.get(), length, data);
  // sum_exp = sum(data) # using simd sum
  float sum_exp = sum_elements(data, length);
  // data /= sum_exp
  for(size_t i = 0; i < length; ++i) {
    data[i] /= sum_exp;
  }
}

REGISTER_UNIT(All2AllSoftmax);

}  // namespace znicz
}  // namespace veles
