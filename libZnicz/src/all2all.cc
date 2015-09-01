/*! @file all2all.cc
 *  @brief "All to all" neural network layer.
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright © 2013 Samsung R&D Institute Russia
 *
 *  @section License
 *  Licensed to the Apache Software Foundation (ASF) under one
 *  or more contributor license agreements.  See the NOTICE file
 *  distributed with this work for additional information
 *  regarding copyright ownership.  The ASF licenses this file
 *  to you under the Apache License, Version 2.0 (the
 *  "License"); you may not use this file except in compliance
 *  with the License.  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an
 *  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 *  KIND, either express or implied.  See the License for the
 *  specific language governing permissions and limitations
 *  under the License.
 */

#include "src/all2all.h"
#include <simd/matrix.h>
#include <simd/memory.h>

namespace veles {
namespace znicz {

All2All::All2All(const std::shared_ptr<Engine>& engine)
    : Unit(engine), include_bias_(true), weights_transposed_(false) {
}

std::vector<std::pair<std::string, std::string>>
All2All::GetParameterDependencies() const noexcept {
  return {{"weights", "weights_transposed"}};
}

void All2All::SetParameter(const std::string& name, const Property& value) {
  if (name == "weights") {
    auto pna = value.get<PackagedNumpyArray>();
    if (!weights_transposed_) {
      weights_ = pna.get<float, 2, true>();
    } else {
      weights_ = pna.get<float, 2, false>();
      weights_.transposed = true;
      std::swap(weights_.shape[0], weights_.shape[1]);
    }
  } else if (name == "bias") {
    bias_ = value.get<PackagedNumpyArray>().get<float, 1>();
  } else if (name == "weights_transposed") {
    weights_transposed_ = !value.get<bool>();
  } else if (name == "include_bias") {
    include_bias_ = value.get<bool>();
  }
}

size_t All2All::OutputSize() const noexcept {
  return weights_.shape[1] * sizeof(float);
}

void All2All::Initialize() {
   Unit::Initialize();
   assert(Parents().size() < 2);
   assert(bias_.shape[0] == weights_.shape[1]);
   assert(weights_.transposed);
}

void All2All::Execute() {
  auto input = Parents().size()?
      reinterpret_cast<float*>(Parents()[0].lock()->output()) :
      reinterpret_cast<const float*>(workflow()->input());
  auto out = reinterpret_cast<float*>(output());
  matrix_multiply_transposed(
      true, input, weights_.data.get_raw(), weights_.shape[0], 1,
      weights_.shape[0], weights_.shape[1], out);
  if (include_bias_) {
    matrix_add(true, out, bias_.data.get_raw(), 1, bias_.shape[0], out);
  }
  ApplyActivationFunction();
}

}  // namespace znicz
}  // namespace veles
