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
#include <cstdlib>
#include <simd/memory.h>
#include <simd/arithmetic.h>
#include "src/all2all_tanh.h"

namespace veles {
namespace znicz {

const std::string All2AllTanh::uuid_ = "b3a2bd5c-3c01-46ef-978a-fef22e008f31";

const std::string& All2AllTanh::Uuid() const noexcept {
  return uuid_;
}

void All2AllTanh::ApplyActivationFunction() const {
  auto out = reinterpret_cast<float*>(output());
  int length = weights_.shape[1];
  real_multiply_scalar(out, length, kScaleX, out);
  for (int i = 0; i < length; i++) {
    // TODO(v.markovtsev): consider adding vectorized tanh calculation to libSimd
    // https://varietyofsound.wordpress.com/2011/02/14/efficient-tanh-computation-using-lamberts-continued-fraction/
    out[i] = std::tanh(out[i]);
  }
  real_multiply_scalar(out, length, kScaleY, out);
}

REGISTER_UNIT(All2AllTanh);

}  // namespace znicz
}  // namespace veles
