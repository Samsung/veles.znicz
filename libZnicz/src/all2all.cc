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

#include <cmath>
#include <memory>
#include <functional>
#include <veles/make_unique.h>
#include <simd/inc/simd/matrix.h>
#include <simd/inc/simd/memory.h>
#include "src/attribute.h"
#include "src/all2all.h"

namespace Veles {
namespace Znicz {

All2All::All2All() : setters_ {
  {"weights", Attribute::GetSetter(&weights_)},
  {"bias", Attribute::GetSetter(&bias_)},
  {"inputs", Attribute::GetSetter(&inputs_)},
  {"outputs", Attribute::GetSetter(&outputs_)}
}, inputs_(0), outputs_(0) {
}

void All2All::SetParameter(const std::string& name,
                           std::shared_ptr<void> value) {
  auto it = setters_.find(name);
  if(it != setters_.end()) {
    it->second(value);
  }
}

/** @brief Execute the neural network layer
 *  @param in Input vector
 *  @param out Output vector
 */
void All2All::Execute(const float* in, float* out) const {
  size_t input_count = InputCount();
  size_t output_count = OutputCount();
  auto tmp = std::uniquify(mallocf(output_count), std::free);
  matrix_multiply(1, weights_.get(), in, input_count,
                  output_count, 1, input_count, tmp.get());
  matrix_add(1, tmp.get(), bias_.get(), 1, output_count, out);
  ApplyActivationFunction(out, outputs_);
}

}  // namespace Znicz
}  // namespace Veles
