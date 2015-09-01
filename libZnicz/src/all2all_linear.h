/*! @file all2all_linear.h
 *  @brief "All to all" neural network layer with linear activation function
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

#ifndef SRC_ALL2ALL_LINEAR_H_
#define SRC_ALL2ALL_LINEAR_H_

#include "src/all2all.h"

#if __GNUC__ >= 4
#pragma GCC visibility push(default)
#endif

namespace veles {
namespace znicz {

/** @brief "All to all" neural network layer with linear activation function
 */
class All2AllLinear : public All2All {
 public:
  explicit All2AllLinear(const std::shared_ptr<Engine>& engine)
      : All2All(engine) {}
  virtual const std::string& Uuid() const noexcept override final;

 protected:
  /** @details Linear activation function, does nothing on the input data:
   *      f(x) = x
   */
  virtual void ApplyActivationFunction() const override final {}

 private:
  static const std::string uuid_;
};

DECLARE_UNIT(All2AllLinear);

}  // namespace znicz
}  // namespace veles

#if __GNUC__ >= 4
#pragma GCC visibility pop
#endif

#endif  // SRC_ALL2ALL_LINEAR_H_
