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

#include <memory>
#include "src/all2all.h"
#include <simd/inc/simd/matrix.h>
#include <simd/inc/simd/memory.h>

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
  size_t input_count = InputCount();
  size_t output_count = OutputCount();
  std::unique_ptr<float, void (*)(void*)> tmp(
      mallocf(output_count), std::free);
  matrix_multiply(1, weights_.get(), in, input_count,
                  output_count, 1, input_count, tmp.get());
  matrix_add(1, tmp.get(), bias_.get(), 1, output_count, out);
  activation_(out);
}

}  // namespace Znicz
}  // namespace Veles
