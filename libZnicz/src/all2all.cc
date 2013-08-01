/*! @file all2all.cc
 *  @brief "All to all" neural network layer.
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include "src/all2all.h"
#include <simd/inc/simd/matrix.h>

namespace Veles {
namespace Znicz {

std::string All2All::Name() const noexcept {
  return "All2All";
}

/** @brief Load unit data from string
 *  @param data %Unit declaration in VELES format
 *  @TODO Define VELES layer loading
 */
void All2All::Load(const std::string& data) {
}

/** @brief Execute the neural network layer
 *  @param in Input vector
 *  @param out Output vector
 */
void All2All::Execute(float* in, float* out) const {
  size_t outputs_count = OutputCount();
  matrix_multiply(1, weights_.get(), in, InputCount(),
                  outputs_count, 1, outputs_count, out);
  for (size_t i = 0; i < outputs_count; ++i) {
    out[i] = neurons_[i]->Execute(out[i] + bias_[i]);
  }
}

}  // namespace Znicz
}  // namespace Veles
