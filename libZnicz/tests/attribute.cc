/*! @file attribute.cc
 *  @brief Attributes manager tests.
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include <gtest/gtest.h>
#include "src/attribute.h"

template<class T>
class AttributeTest : public ::testing::Test {
 protected:
  T variable_;
  std::shared_ptr<T> variable_ptr_;
};

typedef ::testing::Types<int, float> TestTypes;
TYPED_TEST_CASE(AttributeTest, TestTypes);

TYPED_TEST(AttributeTest, Regular) {
  auto setter = Attribute::GetSetter(&this->variable_);
  const TypeParam expected = 42;
  std::shared_ptr<void> parameter = std::make_shared<TypeParam>(expected);
  setter(parameter);
  EXPECT_NEAR(expected, this->variable_, 0.01);
}

TYPED_TEST(AttributeTest, SharedPtr) {
  auto setter = Attribute::GetSetter(&this->variable_ptr_);
  const size_t size = 5;
  TypeParam expected[size] = {1, 42, 24, 199, -5};
  std::shared_ptr<TypeParam> expected_ptr(new TypeParam[size]);
  for(size_t i = 0; i < size; ++i) {
    expected_ptr.get()[i] = expected[i];
  }
  std::shared_ptr<void> parameter(expected_ptr);
  setter(parameter);
  for(size_t i = 0; i < size; ++i) {
    ASSERT_NEAR(expected[i], this->variable_ptr_.get()[i], 0.01);
  }
}

TYPED_TEST(AttributeTest, String) {
  auto setter = Attribute::GetSetterString(&this->variable_);
  TypeParam expected = 31415;
  std::string expected_string = "31415";
  std::shared_ptr<void> parameter = std::make_shared<std::string>(
      expected_string);
  setter(parameter);
  EXPECT_NEAR(expected, this->variable_, 0.01);
}

TYPED_TEST(AttributeTest, StringFormat) {
  auto setter = Attribute::GetSetterString(&this->variable_);
  std::string expected_string = "314q15";
  std::shared_ptr<void> parameter = std::make_shared<std::string>(
      expected_string);
  ASSERT_THROW(setter(parameter), std::stringstream::failure);
}

#include "tests/google/src/gtest_main.cc"
