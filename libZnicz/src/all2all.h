/*! @file all2all.h
 *  @brief "All to all" neural network layer
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

#ifndef SRC_ALL2ALL_H_
#define SRC_ALL2ALL_H_

#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include <veles/veles.h>

template <class T> class All2AllTest;

namespace veles {
namespace znicz {

/** @brief "All to all" neural network layer
 */
class All2All : public Unit {
 public:
  explicit All2All(const std::shared_ptr<Engine>& engine);
  virtual void SetParameter(
      const std::string& name, const Property& value) override;

  virtual std::vector<std::pair<std::string, std::string>>
  GetParameterDependencies() const noexcept override;

  virtual size_t OutputSize() const noexcept override final;

  virtual void Initialize() override;

 protected:
  template <class T> friend class ::All2AllTest;

  virtual void Execute() override;
  /** @brief Activation function used by the neural network layer.
   *  @param data Vector to be transformed
   *  @param length Number of elements in the data vector
   */
  virtual void ApplyActivationFunction() const = 0;

  /** @brief Weights matrix
   */
  NumpyArray<float, 2> weights_;
  /** @brief Bias vector
   */
  NumpyArray<float, 1> bias_;
  bool include_bias_;
  bool weights_transposed_;
};

}  // namespace znicz
}  // namespace veles

#endif  // SRC_ALL2ALL_H_
