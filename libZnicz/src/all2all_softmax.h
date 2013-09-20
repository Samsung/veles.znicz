/*! @file all2all_softmax.h
 *  @brief "All to all" neural network layer with Softmax activation function
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef SRC_ALL2ALL_SOFTMAX_H_
#define SRC_ALL2ALL_SOFTMAX_H_

#include "src/all2all.h"

#if __GNUC__ >= 4
#pragma GCC visibility push(default)
#endif

namespace Veles {
namespace Znicz {

/** @brief "All to all" neural network layer with Softmax activation function
 */
class All2AllSoftmax : public All2All {
 public:
  virtual std::string Name() const noexcept override final;

 protected:
  virtual void ApplyActivationFunction(float* data,
                                       size_t length) const override final;
};

DECLARE_UNIT(All2AllSoftmax);

}  // namespace Znicz
}  // namespace Veles

#if __GNUC__ >= 4
#pragma GCC visibility pop
#endif

#endif  // SRC_ALL2ALL_SOFTMAX_H_
