/*! @file all2all_linear.h
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

#ifndef TESTS_ALL2ALL_LINEAR_H_
#define TESTS_ALL2ALL_LINEAR_H_

#include "tests/all2all.h"

class All2AllLinear : public All2AllTest {
 protected:
  All2AllLinear() : All2AllTest("All2All") {
  }
  virtual void SetUp() override;
};

class All2AllLinearSquare : public All2AllLinear {
 protected:
  static const size_t kCount;

  virtual void SetUp() override;
};

#endif  // TESTS_ALL2ALL_H_
