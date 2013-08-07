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
#include <veles/unit.h>

namespace Veles {
namespace Znicz {

/** @brief "All to all" neural network layer. */
class All2All : public Unit {
 public:
    All2All() noexcept : inputs_(0), outputs_(0) {
  }
  virtual ~All2All() noexcept {
  }
  virtual std::string Name() const noexcept override final;

  virtual void Load(const std::string& data) override final;
  virtual void Execute(float* in, float* out) const override final;
  virtual size_t InputCount() const noexcept override final {
    return inputs_;
  }
  virtual size_t OutputCount() const noexcept override final {
    return outputs_;
  }

 private:
  std::unique_ptr<float[]> weights_;
  std::unique_ptr<float[]> bias_;
  std::function<void (float*)> activation_;
  size_t inputs_;
  size_t outputs_;
};

}  // namespace Znicz
}  // namespace Veles

#endif  // SRC_ALL2ALL_H_
