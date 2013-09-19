/*! @file all2all_softmax.h
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

#ifndef TESTS_ALL2ALL_SOFTMAX_H_
#define TESTS_ALL2ALL_SOFTMAX_H_

#include "tests/all2all.h"

class All2AllSoftmax : public All2AllTest {
 protected:
  All2AllSoftmax() : All2AllTest("All2AllSoftmax") {
  }
  virtual void SetUp() override;
};

#endif  // TESTS_ALL2ALL_SOFTMAX_H_
