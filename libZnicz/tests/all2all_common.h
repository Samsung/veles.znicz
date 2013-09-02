/*! @file all2all_common.h
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

#ifndef TESTS_ALL2ALL_COMMON_H_
#define TESTS_ALL2ALL_COMMON_H_

#include <tuple>
#include "tests/all2all.h"

class All2AllEmptyConstruction :
    public All2AllTest, public ::testing::WithParamInterface<const char*> {
 protected:
  All2AllEmptyConstruction() : All2AllTest(GetParam()) {
  }
  virtual void SetUp() override {
    Initialize(kInputs, kOutputs);
  }
};

class All2AllCommon : public All2AllTest,
                      public ::testing::WithParamInterface<
                        std::tuple<const char*, size_t, size_t>> {
 protected:
  All2AllCommon() : All2AllTest(std::get<0>(GetParam())) {
  }
  virtual void SetUp() override {
    size_t inputs = 0;
    size_t outputs = 0;
    std::tie(std::ignore, inputs, outputs) = GetParam();
    Initialize(inputs, outputs);
  }
};

#endif  // TESTS_ALL2ALL_COMMON_H_
