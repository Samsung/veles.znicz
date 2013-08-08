/*! @file all2all.h
 *  @brief "All to all" neural network layer.
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

namespace Veles {
namespace Znicz {

/** @brief "All to all" neural network layer. */
class All2All : public Unit {
 public:
  /** @brief Constructs an empty unit
   */
  All2All() noexcept;
  virtual ~All2All() noexcept {
  }
  virtual std::string Name() const noexcept override final;
  virtual void SetParameter(const std::string& name,
                            std::shared_ptr<void> value) override final;
  virtual void Execute(float* in, float* out) const override final;
  virtual size_t InputCount() const noexcept override final {
    return inputs_;
  }
  virtual size_t OutputCount() const noexcept override final {
    return outputs_;
  }

 private:
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

  /** @brief Weights matrix
   */
  std::shared_ptr<float> weights_;
  /** @brief Bias vector
   */
  std::shared_ptr<float> bias_;
  /** @brief Activation function, that applies in-place transform
   *  to the output vector
   */
  std::function<void (float*, size_t)> activation_;
  /** @brief Parameter name to Parameter setter map
   */
  std::unordered_map<std::string, std::function<void (std::shared_ptr<void>)>>
    setters_;
  /** @brief Number of inputs
   */
  size_t inputs_;
  /** Number of outputs
   */
  size_t outputs_;
};

}  // namespace Znicz
}  // namespace Veles

#endif  // SRC_ALL2ALL_H_
