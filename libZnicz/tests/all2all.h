/*! @file all2all.h
 *  @brief "All to all" base test fixture
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright © 2013 Samsung R&D Institute Russia
 *
 *  @section License
 *  Licensed to the Apache Software Foundation (ASF) under one
 *  or more contributor license agreements.  See the NOTICE file
 *  distributed with this work for additional information
 *  regarding copyright ownership.  The ASF licenses this file
 *  to you under the Apache License, Version 2.0 (the
 *  "License"); you may not use this file except in compliance
 *  with the License.  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an
 *  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 *  KIND, either express or implied.  See the License for the
 *  specific language governing permissions and limitations
 *  under the License.
 */

#ifndef TESTS_ALL2ALL_H_
#define TESTS_ALL2ALL_H_

#define GTEST_HAS_TR1_TUPLE 1

#include <cmath>
#include <memory>
#include <gtest/gtest.h>
#include <veles/veles.h>
#include <simd/memory.h>

class DummyUnit : public veles::Unit {
 public:
  DummyUnit() : veles::Unit(nullptr) {}

  virtual const std::string& Uuid() const noexcept override {
    return uuid_;
  }

  virtual void SetParameter(const std::string&,
                            const veles::Property&) override {
  }

  virtual size_t OutputSize() const noexcept override {
    return 0;
  }

 protected:
  virtual void Execute() override {
  }

 private:
  static const std::string uuid_;
};

const std::string DummyUnit::uuid_ = "uuid";

template <class T>
class All2AllTest :
    public ::testing::Test,
    public veles::DefaultLogger<T, veles::Logger::COLOR_RED> {
 public:
  All2AllTest<T>() : height_(0), width_(0), unit_(new T(nullptr)),
                     parent_(new DummyUnit()) {}
  using Parent = All2AllTest<T>;

  void Initialize() {
    unit_->weights_ = veles::NumpyArray<float, 2>();
    unit_->weights_.shape[0] = height_;
    unit_->weights_.shape[1] = width_;
    unit_->weights_.transposed = true;
    unit_->weights_.data = veles::shared_array<float>(weights_, height_ * width_);
    unit_->include_bias_ = true;
    unit_->bias_.shape[0] = width_;
    unit_->bias_.data = veles::shared_array<float>(bias_, width_);
    output_ = std::shared_ptr<float>(mallocf(width_), std::free);
    unit_->set_output(output_.get());
    input_ = std::shared_ptr<float>(mallocf(height_), std::free);
    parent_->set_output(input_.get());
    unit_->LinkFrom(parent_);
  }

  void Verify(std::initializer_list<float> input,
              std::initializer_list<float> expected) {
    assert(input.size() == height_);
    std::copy(input.begin(), input.end(), input_.get());
    unit_->Initialize();
    unit_->Execute();
    int i = 0;
    for (auto ex : expected) {
      EXPECT_NEAR(ex, output_.get()[i++], std::fabs(ex) / 1000000) << i - 1;
    }
  }

 protected:
  static std::shared_ptr<float> CreateFloatArray(
      std::initializer_list<float> init) {
    auto ptr = std::shared_ptr<float>(mallocf(init.size()), std::free);
    std::copy(init.begin(), init.end(), ptr.get());
    return ptr;
  }

  size_t height_;
  size_t width_;
  std::shared_ptr<float> weights_;
  std::shared_ptr<float> bias_;
  std::shared_ptr<float> input_;
  std::shared_ptr<float> output_;

 private:
  std::shared_ptr<T> unit_;
  std::shared_ptr<DummyUnit> parent_;
};

#endif  // TESTS_ALL2ALL_H_
