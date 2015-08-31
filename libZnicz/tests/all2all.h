/*! @file all2all.h
 *  @brief "All to all" base test fixture
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef TESTS_ALL2ALL_H_
#define TESTS_ALL2ALL_H_

#define GTEST_HAS_TR1_TUPLE 1

#include <memory>
#include <gtest/gtest.h>
#include <veles/unit.h>
#include <veles/unit_factory.h>
#include <simd/memory.h>
#include "tests/common_test.h"

class All2AllTest : public CommonTest {
 protected:
  static const size_t kInputs;
  static const size_t kOutputs;

  All2AllTest(const std::string& name)
    : unit_(nullptr), name_(name) {
  }

  void Initialize(size_t inputs, size_t outputs,
                  float* weights = nullptr, float* bias = nullptr) {
    InitializeUnit(unit(), inputs, outputs, weights, bias);
    input_ = CreateFloatArray(inputs, kValueInputInit);
    output_ = CreateFloatArray(outputs, kValueOutputInit);
  }

  template<class InputIterator>
  void TestExecution(InputIterator expected_begin,
                     InputIterator expected_end) {
    unit()->Execute(input().get(), output().get());
    int i = 0;
    for (InputIterator it = expected_begin; it != expected_end; ++it, ++i) {
      ASSERT_NEAR(*it, output().get()[i], 0.01)
        << "i = " << i << std::endl;
    }
  }

  std::shared_ptr<veles::Unit> unit() const {
    if (!unit_) {
      unit_ = CreateUnit(name_);
    }
    return unit_;
  }

  std::shared_ptr<float> input() {
    return input_;
  }

  std::shared_ptr<float> output() {
    return output_;
  }

  std::string name() const {
    return name_;
  }

 private:
  mutable std::shared_ptr<veles::Unit> unit_;
  std::shared_ptr<float> input_;
  std::shared_ptr<float> output_;
  std::string name_;
};

const size_t All2AllTest::kInputs = 3;
const size_t All2AllTest::kOutputs = 2;

#endif  // TESTS_ALL2ALL_H_
