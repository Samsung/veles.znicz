/*! @file all2all.h
 *  @brief "All to all" neural network layer
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef SRC_ALL2ALL_H_
#define SRC_ALL2ALL_H_

#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include <veles/unit.h>
#include <veles/unit_factory.h>

namespace veles {
namespace znicz {

/** @brief "All to all" neural network layer
 */
class All2All : public Unit {
 public:
  All2All();
  virtual void SetParameter(const std::string& name,
                            std::shared_ptr<void> value) override final;
  virtual void Execute(const float* in, float* out) const override final;
  virtual size_t InputCount() const noexcept override final {
    size_t outputs = OutputCount();
    return outputs ? weights_length_ / outputs : 0;
  }
  virtual size_t OutputCount() const noexcept override final {
    return bias_length_;
  }

 protected:
  /** @brief Activation function used by the neural network layer.
   *  @param data Vector to be transformed
   *  @param length Number of elements in the data vector
   */
  virtual void ApplyActivationFunction(float* data, size_t length) const = 0;

 private:
  /** @brief Weights matrix
   */
  std::shared_ptr<float> weights_;
  /** @brief Bias vector
   */
  std::shared_ptr<float> bias_;
  /** @brief Parameter name to Parameter setter map
   */
  std::unordered_map<std::string, std::function<void (std::shared_ptr<void>)>>
    setters_;
  /** @brief Number of elements in the weights matrix.
   *  @details Number of inputs is computed using this length and outputs count.
   */
  size_t weights_length_;
  /** @brief Number of elements in the bias vector.
   *  @details Equals to the number of outputs.
   */
  size_t bias_length_;
};

}  // namespace znicz
}  // namespace veles

#endif  // SRC_ALL2ALL_H_
