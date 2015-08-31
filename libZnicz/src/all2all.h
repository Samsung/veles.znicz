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
#include <veles/veles.h>

namespace veles {
namespace znicz {

/** @brief "All to all" neural network layer
 */
class All2All : public Unit {
 public:
  explicit All2All(const std::shared_ptr<Engine>& engine);
  virtual void SetParameter(
      const std::string& name, const Property& value) override;

  virtual std::vector<std::pair<std::string, std::string>>
  GetParameterDependencies() const noexcept override;

  virtual size_t OutputSize() const noexcept override final;

  virtual void Initialize() override;

 protected:
  virtual void Execute() override;
  /** @brief Activation function used by the neural network layer.
   *  @param data Vector to be transformed
   *  @param length Number of elements in the data vector
   */
  virtual void ApplyActivationFunction() const = 0;

  /** @brief Weights matrix
   */
  NumpyArray<float, 2> weights_;
  /** @brief Bias vector
   */
  NumpyArray<float, 1> bias_;
  bool include_bias_;
  bool weights_transposed_;
};

}  // namespace znicz
}  // namespace veles

#endif  // SRC_ALL2ALL_H_
