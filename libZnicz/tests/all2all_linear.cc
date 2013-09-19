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

#include "tests/all2all_linear.h"
#include <vector>
#include "src/all2all_linear.h"

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
  size_t count = GetParam();
  Initialize(count, count);
}

TEST_P(All2AllLinearSquare, ExecutionIdentity) {
  size_t count = GetParam();
  auto weights = CreateFloatArray(count * count);
  for (size_t i = 0; i < count; ++i) {
    weights.get()[(count + 1) * i] = kValueOne;
  }
  unit()->SetParameter("weights", weights);
  std::vector<float> expected(count, kValueInputInit);
  TestExecution(expected.begin(), expected.end());
}

TEST_P(All2AllLinearSquare, ExecutionBias) {
  size_t count = GetParam();
  auto bias = CreateFloatArray(count * count, kValueOther);
  unit()->SetParameter("bias", bias);
  std::vector<float> expected(count, kValueOther);
  TestExecution(expected.begin(), expected.end());
}

INSTANTIATE_TEST_CASE_P(All2AllLinearSquareTests, All2AllLinearSquare,
                        ::testing::Values(1, 5, 199));

#include "tests/google/src/gtest_main.cc"
