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

#include <memory>
#include <gtest/gtest.h>
#include <veles/unit.h>
#include <veles/unit_factory.h>
#include <simd/inc/simd/memory.h>

class All2AllTest : public ::testing::Test {
 protected:
  static const float kValueZero;
  static const float kValueOne;
  static const float kValueInputInit;
  static const float kValueOutputInit;
  static const float kValueOther;
  static const size_t kInputs;
  static const size_t kOutputs;

  All2AllTest(const std::string& name)
    : unit_(nullptr), name_(name) {
  }

  virtual ~All2AllTest() = default;

  static std::shared_ptr<Veles::Unit> CreateUnit(const std::string& name) {
    return Veles::UnitFactory::Instance()[name]();
  }

  std::shared_ptr<float> CreateFloatArray(size_t count,
                                          float initializer = kValueZero) {
    auto ptr = std::shared_ptr<float>(mallocf(count), std::free);
    memsetf(ptr.get(), count, initializer);
    return ptr;
  }

  void Initialize(size_t inputs, size_t outputs,
                  float* weights = nullptr, float* bias = nullptr) {
    auto weights_array = CreateFloatArray(inputs * outputs);
    auto bias_array = CreateFloatArray(outputs);
    if (weights) {
      memcpy(weights_array.get(), weights, inputs * outputs * sizeof(float));
    }
    if (bias) {
      memcpy(bias_array.get(), bias, outputs * sizeof(float));
    }
    input_ = CreateFloatArray(inputs, kValueInputInit);
    output_ = CreateFloatArray(outputs, kValueOutputInit);
    unit()->SetParameter("weights", weights_array);
    unit()->SetParameter("bias", bias_array);
    unit()->SetParameter("inputs", std::make_shared<size_t>(inputs));
    unit()->SetParameter("outputs", std::make_shared<size_t>(outputs));
  }

  template<class InputIterator>
  void TestExecution(InputIterator expected_begin,
                     InputIterator expected_end) {
    unit()->Execute(input().get(), output().get());
    int i = 0;
    for (InputIterator it = expected_begin; it != expected_end; ++it, ++i) {
      ASSERT_NEAR(*it, output().get()[i], 0.01);
    }
  }

  std::shared_ptr<Veles::Unit> unit() const {
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
  mutable std::shared_ptr<Veles::Unit> unit_;
  std::shared_ptr<float> input_;
  std::shared_ptr<float> output_;
  std::string name_;
};

const float All2AllTest::kValueZero = 0;
const float All2AllTest::kValueOne = 1;
const float All2AllTest::kValueInputInit = 42.42;
const float All2AllTest::kValueOutputInit = 412.31415;
const float All2AllTest::kValueOther = 156.27172;
const size_t All2AllTest::kInputs = 3;
const size_t All2AllTest::kOutputs = 2;

#endif  // TESTS_ALL2ALL_H_
