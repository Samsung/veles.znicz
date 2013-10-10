/*! @file attribute.h
 *  @brief Unit attributes manager
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

/*!
 * @mainpage Neural network realization.
 * \image html neural-network.jpg
 *
 */

#ifndef SRC_ATTRIBUTE_H_
#define SRC_ATTRIBUTE_H_

#include <memory>
#include <string>
#include <sstream>
#include <functional>


/** @brief Unit attributes manager, used to generate parameter assignment
 *  functions taking shared_ptr<void> values.
 */
class Attribute {
 public:
  /** @brief Constructs a function that converts shared_ptr<void> to pointer
   *  to the desired type and assigns pointed value to the provided variable
   *  using full copy.
   *  @param data Variable, setter for which is constructed
   */
  template<class T>
  static std::function<void (std::shared_ptr<void>)>
  GetSetter(T* data) {
    return [data](std::shared_ptr<void> value) {
      *data = *std::static_pointer_cast<T>(value);
    };
  }

  /** @brief Constructs a function that takes shared_ptr<void> and assigns it
   *  to the provided variable without doing full copy of contents.
   *  @param data Variable of shared_ptr type, setter for which is constructed.
   */
  template<class T>
  static std::function<void (std::shared_ptr<void>)>
  GetSetter(std::shared_ptr<T>* data) {
    return [data](std::shared_ptr<void> value) {
      *data = std::static_pointer_cast<T>(value);
    };
  }

  /** @brief Constructs a function that converts shared_ptr<void> to pointer
   *  to string and uses lexical conversion to assign it to the provided
   *  variable.
   *  @param data Variable, setter for which is constructed
   */
  template<class T>
  static std::function<void (std::shared_ptr<void>)>
  GetSetterString(T* data) {
    return [data](std::shared_ptr<void> value) {
      *data = lexical_cast<T>(*std::static_pointer_cast<std::string>(value));
    };
  }

  /** @brief Converts the string to the selected type.
   *  @tparam T Destination type.
   *  @param str Input string.
   */
  template <typename T>
  static T lexical_cast(const std::string& str)
  {
      T value;
      std::istringstream iss;
      iss.exceptions(std::istringstream::failbit | std::istringstream::badbit);
      iss.str(str);
      iss >> value;
      if(!iss.eof()) {
        throw std::stringstream::failure("lexical_cast");
      }
      return value;
  }
};

#endif  // SRC_ATTRIBUTE_H_
