/*! @file all2all_common.cc
 *  @brief Parameterized tests for All2All units
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
#include "src/all2all_tanh.h"
#include "tests/all2all_common.h"

TEST_P(All2AllCommon, EmptyConstruction) {
  std::shared_ptr<Veles::Unit> unit(CreateUnit(name()));
  EXPECT_EQ(name(), unit->Name());
  EXPECT_EQ(kValueZero, unit->InputCount());
  EXPECT_EQ(kValueZero, unit->OutputCount());
}

TEST_P(All2AllCommon, Construction) {
  EXPECT_EQ(name(), unit()->Name());
  EXPECT_EQ(kInputs, unit()->InputCount());
  EXPECT_EQ(kOutputs, unit()->OutputCount());
}

TEST_P(All2AllCommon, ExecutionZeroWeightsBias) {
  unit()->Execute(input().get(), output().get());
  for(size_t i = 0; i < kOutputs; ++i) {
    EXPECT_EQ(kValueZero, output().get()[i]);
  }
}

INSTANTIATE_TEST_CASE_P(All2AllCommonTests, All2AllCommon,
                        ::testing::Values("All2All", "All2AllTanh"));

GTEST_API_ int main(int argc, char **argv) {
  REFERENCE_UNIT(Veles::Znicz, All2AllLinear);
  REFERENCE_UNIT(Veles::Znicz, All2AllTanh);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
