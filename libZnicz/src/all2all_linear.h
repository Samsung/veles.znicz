/*! @file all2all_linear.h
 *  @brief "All to all" neural network layer with linear activation function
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef SRC_ALL2ALL_LINEAR_H_
#define SRC_ALL2ALL_LINEAR_H_

#include "src/all2all.h"

#if __GNUC__ >= 4
#pragma GCC visibility push(default)
#endif

namespace veles {
namespace znicz {

/** @brief "All to all" neural network layer with linear activation function
 */
class All2AllLinear : public All2All {
 public:
  virtual std::string Name() const noexcept override final;

 protected:
  /** @details Linear activation function, does nothing on the input data:
   *      f(x) = x
   */
  virtual void ApplyActivationFunction(float*,  size_t) const override final {
  }
};

DECLARE_UNIT(All2AllLinear);

}  // namespace znicz
}  // namespace veles

#if __GNUC__ >= 4
#pragma GCC visibility pop
#endif

#endif  // SRC_ALL2ALL_LINEAR_H_
