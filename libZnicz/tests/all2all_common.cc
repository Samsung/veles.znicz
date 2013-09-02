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

#include <veles/unit_factory.h>
#include "src/all2all_linear.h"
#include "src/all2all_tanh.h"
#include "tests/all2all_common.h"

TEST_P(All2AllEmptyConstruction, EmptyConstruction) {
  std::shared_ptr<Veles::Unit> unit(CreateUnit(name()));
  EXPECT_EQ(name(), unit->Name());
  EXPECT_EQ(kValueZero, unit->InputCount());
  EXPECT_EQ(kValueZero, unit->OutputCount());
}

TEST_P(All2AllCommon, Construction) {
  size_t inputs = 0;
  size_t outputs = 0;
  std::tie(std::ignore, inputs, outputs) = GetParam();
  EXPECT_EQ(name(), unit()->Name());
  EXPECT_EQ(inputs, unit()->InputCount());
  EXPECT_EQ(outputs, unit()->OutputCount());
}

TEST_P(All2AllCommon, ExecutionZeroWeightsBias) {
  size_t outputs = 0;
  std::tie(std::ignore, std::ignore, outputs) = GetParam();
  unit()->Execute(input().get(), output().get());
  for (size_t i = 0; i < outputs; ++i) {
    ASSERT_EQ(kValueZero, output().get()[i])
        << "i = " << i << std::endl;
  }
}

INSTANTIATE_TEST_CASE_P(All2AllEmptyConstructionTests, All2AllEmptyConstruction,
                        ::testing::Values("All2All", "All2AllTanh"));


INSTANTIATE_TEST_CASE_P(
    All2AllCommonTests, All2AllCommon,
    ::testing::Combine(
        ::testing::Values("All2All", "All2AllTanh"),
        ::testing::Values(1, 5, 299),
        ::testing::Values(1, 10, 299)));

#include "tests/google/src/gtest_main.cc"
