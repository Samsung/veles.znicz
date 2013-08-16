/*! @file all2all_tanh.h
 *  @brief "All to all" neural network layer with Tanh activation function
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef SRC_ALL2ALL_TANH_H_
#define SRC_ALL2ALL_TANH_H_

#include "src/all2all.h"

namespace Veles {
namespace Znicz {

/** @brief "All to all" neural network layer with Tanh activation function
 */
class All2AllTanh : public All2All {
 public:
  virtual std::string Name() const noexcept override final;

 protected:
  /** @brief Activation function used by the neural network layer.
   *  @param data Vector to be transformed
   *  @param length Number of elements in the data vector
   *  @details Tanh activation function:
   *      f(x) = 1.7159 * tanh(0.6666 * x)
   */
  virtual void ApplyActivationFunction(float* data,
                                       size_t length) const override final;

 private:
  /** @brief Scale of the input vector
   */
  static constexpr float kScaleX = 0.6666;
  /** @brief Scale of the output vector
   */
  static constexpr float kScaleY = 1.7159;
};

}  // namespace Znicz
}  // namespace Veles

#endif  // SRC_ALL2ALL_TANH_H_
