/*! @file all2all_tanh.h
 *  @brief "All to all" neural network layer with Tanh activation function
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

#ifndef SRC_ALL2ALL_TANH_H_
#define SRC_ALL2ALL_TANH_H_

#include "src/all2all.h"

#if __GNUC__ >= 4
#pragma GCC visibility push(default)
#endif

namespace veles {
namespace znicz {

/** @brief "All to all" neural network layer with Tanh activation function
 */
class All2AllTanh : public All2All {
 public:
  explicit All2AllTanh(const std::shared_ptr<Engine>& engine)
      : All2All(engine) {}
  virtual const std::string& Uuid() const noexcept override final;

 protected:
  /** @brief Activation function used by the neural network layer.
   *  @param data Vector to be transformed
   *  @param length Number of elements in the data vector
   *  @details Tanh activation function:
   *      f(x) = 1.7159 * tanh(0.6666 * x)
   */
  virtual void ApplyActivationFunction() const override final;

 private:
  /** @brief Scale of the input vector
   */
  static constexpr float kScaleX = 0.6666;
  /** @brief Scale of the output vector
   */
  static constexpr float kScaleY = 1.7159;
  static const std::string uuid_;
};

DECLARE_UNIT(All2AllTanh);

}  // namespace znicz
}  // namespace veles

#if __GNUC__ >= 4
#pragma GCC visibility pop
#endif

#endif  // SRC_ALL2ALL_TANH_H_
