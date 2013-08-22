/*! @file all2all_linear.cc
 *  @brief "All to all" unit with linear activation function test
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
#include "src/all2all_linear.h"
#include "tests/all2all_linear.h"

const size_t All2AllLinearSquare::kCount = 5;

void All2AllLinear::SetUp() {
  float weights[kInputs * kOutputs] = {1, 2.1, 33,
                                         4, 55, 6 };
  float bias[kOutputs] = {433, 42.9};
  Initialize(kInputs, kOutputs, weights, bias);
}

TEST_F(All2AllLinear, Execution) {
  std::vector<float> expected = {1964.362, 2800.2};
  TestExecution(expected.begin(), expected.end());
}

void All2AllLinearSquare::SetUp() {
  Initialize(kCount, kCount);
}

TEST_F(All2AllLinearSquare, ExecutionIdentity) {
  auto weights = CreateFloatArray(kCount * kCount);
  for (size_t i = 0; i < kCount; ++i) {
    weights.get()[(kCount + 1) * i] = kValueOne;
  }
  unit()->SetParameter("weights", weights);
  std::vector<float> expected(kCount, kValueInputInit);
  TestExecution(expected.begin(), expected.end());
}

TEST_F(All2AllLinearSquare, ExecutionBias) {
  auto bias = CreateFloatArray(kCount * kCount, kValueOther);
  unit()->SetParameter("bias", bias);
  std::vector<float> expected(kCount, kValueOther);
  TestExecution(expected.begin(), expected.end());
}

#include "tests/google/src/gtest_main.cc"
