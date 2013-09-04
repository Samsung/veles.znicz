/*! @file functional_test.cc
 *  @brief Functional workflow test using Znicz units.
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include "src/all2all_tanh.h"
#include "src/all2all_linear.h"
#include "tests/functional_test.h"

const size_t WorkflowTest::kInputsLinear = 4;
const size_t WorkflowTest::kOutputsLinear = 3;
const size_t WorkflowTest::kOutputsTanh = 2;

void WorkflowTest::SetUp() {
  float weights_linear[kInputsLinear * kOutputsLinear] = {
        1,    2.3, -4.2, 0.6,
        0.8, -2.1,  1.3, 2.8,
        0,    1.9, -0.1, 3 };
  float bias_linear[kOutputsLinear] = { 13, -118, -205 };
  float weights_tanh[kOutputsLinear * kOutputsTanh] = {
       1,  2.3, 4.2,
      -2, -0.8, 0 };
  float bias_tanh[kOutputsTanh] = { 4.5, 1 };
  auto unit_linear = CreateUnit("All2All");
  InitializeUnit(unit_linear, kInputsLinear, kOutputsLinear,
                 weights_linear, bias_linear);
  auto unit_tanh = CreateUnit("All2AllTanh");
  InitializeUnit(unit_tanh, kOutputsLinear, kOutputsTanh,
                 weights_tanh, bias_tanh);
  workflow()->Add(unit_linear);
  workflow()->Add(unit_tanh);
}

TEST_F(WorkflowTest, Functional) {
  workflow()->Execute(input().get(), input().get() + kInputsLinear,
                      output().get());
  float expected[kOutputsTanh] = { 0.789, -0.186 };
  for (size_t i = 0; i < kOutputsTanh; ++i) {
      EXPECT_NEAR(expected[i], output().get()[i], 0.01);
  }
}

#include "tests/google/src/gtest_main.cc"
