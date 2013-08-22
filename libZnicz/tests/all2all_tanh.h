/*! @file all2all_tanh.h
 *  @brief "All to all" unit with Tanh activation function test
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef TESTS_ALL2ALL_TANH_H_
#define TESTS_ALL2ALL_TANH_H_

#include "tests/all2all.h"

class All2AllTanh : public All2AllTest {
 protected:
  All2AllTanh() : All2AllTest("All2AllTanh") {
  }
  virtual void SetUp() override;
};

#endif  // TESTS_ALL2ALL_TANH_H_
