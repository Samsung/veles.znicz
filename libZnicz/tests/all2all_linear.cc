/*! @file all2all_linear.cc
 *  @brief "All to all" unit with linear activation function test
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

#include "tests/all2all_linear.h"
#include <vector>
#include "src/all2all_linear.h"

namespace veles {

namespace znicz {

TEST_F(All2AllLinearTest, Execution) {
  height_ = 5;
  width_ = 3;
  // transposed
  weights_ = CreateFloatArray({ 1, 0, 2, 1, -1,
                                3, 1, 0, 2,  3,
                               -1, 2, 0, 1,  3});
  bias_ = CreateFloatArray({ 10, -10, 5 });
  Initialize();
  Verify({ 1, 2, 3, 2, 1 }, { 18, 2, 13 });
}

}

}

#include "tests/google/src/gtest_main.cc"
