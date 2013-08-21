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


#include <veles/unit_registry.h>
#include "src/all2all_linear.h"
#include "tests/all2all_linear.h"

const size_t All2AllLinearSquare::kCount = 5;

TEST_F(All2AllLinear, Execution) {
  float weights_f[kInputs * kOutputs] = {1, 2.1, 33,
                                         4, 55, 6 };
  float bias_f[kOutputs] = {433, 42.9};
  float expected[kOutputs] = {1964.362, 2800.2};
  auto weights = CreateFloatArray(kInputs * kOutputs);
  auto bias = CreateFloatArray(kOutputs);
  memcpy(weights.get(), weights_f, sizeof(weights_f));
  memcpy(bias.get(), bias_f, sizeof(bias_f));
  unit()->SetParameter("weights", weights);
  unit()->SetParameter("bias", bias);
  unit()->Execute(input().get(), output().get());
  for(size_t i = 0; i < kOutputs; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output().get()[i]);
  }
}

TEST_F(All2AllLinearSquare, ExecutionIdentity) {
  auto weights = CreateFloatArray(kCount * kCount);
  for(size_t i = 0; i < kCount; ++i) {
    weights.get()[(kCount + 1) * i] = kValueOne;
  }
  unit()->SetParameter("weights", weights);
  unit()->Execute(input().get(), output().get());
  for(size_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(kValueInputInit, output().get()[i]);
  }
}

TEST_F(All2AllLinearSquare, ExecutionBias) {
  auto bias = CreateFloatArray(kCount * kCount, kValueOther);
  unit()->SetParameter("bias", bias);
  unit()->Execute(input().get(), output().get());
  for(size_t i = 0; i < kCount; ++i) {
    EXPECT_EQ(kValueOther, output().get()[i]);
  }
}

GTEST_API_ int main(int argc, char **argv) {
  REFERENCE_UNIT(Veles::Znicz, All2AllLinear);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
