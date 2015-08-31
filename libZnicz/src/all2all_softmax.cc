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
#include <simd/arithmetic.h>
#include <veles/make_unique.h>
#include "src/all2all_softmax.h"

namespace veles {
namespace znicz {

const std::string All2AllSoftmax::uuid_ = "420219fc-3e1a-45b1-87f8-aaa0c1540de4";

const std::string& All2AllSoftmax::Uuid() const noexcept {
  return uuid_;
}

void All2AllSoftmax::ApplyActivationFunction() const {
  auto out = reinterpret_cast<float*>(output());
  int length = weights_.shape[1];
  float max = *std::max_element(out, out + length);
  add_to_all(out, length, -max, out);
  exp_psv(true, out, length, out);
  float sum_exp = sum_elements(out, length);
  real_multiply_scalar(out, length, 1 / sum_exp, out);
}

REGISTER_UNIT(All2AllSoftmax);

}  // namespace znicz
}  // namespace veles
