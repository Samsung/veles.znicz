/*! @file all2all_softmax.cc
 *  @brief "All to all" unit with Softmax activation function test
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
#include "tests/all2all_softmax.h"
#include "src/all2all_softmax.h"

void All2AllSoftmax::SetUp() {
  float weights[kInputs * kOutputs] = {1, 2.1, 33,
                                       4, 55, 6 };
  float bias[kOutputs] = {-1531.362 - 0.5, -2757.3 + 0.3};
  Initialize(kInputs, kOutputs, weights, bias);
}

TEST_F(All2AllSoftmax, Execution) {
  std::vector<float> expected = {0,31, 0,67};
  TestExecution(expected.begin(), expected.end());
}

#include "tests/google/src/gtest_main.cc"

