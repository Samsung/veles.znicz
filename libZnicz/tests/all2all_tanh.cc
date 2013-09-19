/*! @file all2all_tanh.cc
 *  @brief "All to all" unit with Tanh activation function test
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include <vector>
#include "tests/all2all_tanh.h"
#include "src/all2all_tanh.h"

void All2AllTanh::SetUp() {
  float weights[kInputs * kOutputs] = {1, 2.1, 33,
                                       4, 55, 6 };
  float bias[kOutputs] = {-1531.462, -2756.6};
  Initialize(kInputs, kOutputs, weights, bias);
}

TEST_F(All2AllTanh, Execution) {
  std::vector<float> expected = {-0.1142, 0.7472};
  TestExecution(expected.begin(), expected.end());
}

#include "tests/google/src/gtest_main.cc"
