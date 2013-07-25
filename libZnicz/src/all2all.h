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

#ifndef INC_ALL2ALL_H_
#define INC_ALL2ALL_H_

#include <string>
#include <memory>
#include <vector>
#include <veles/unit.h>
#include "src/neuron.h"

namespace Veles {
namespace Znicz {

/** @brief "All to all" neural network layer. */
class All2All : public Unit {
 public:
  All2All() noexcept : inputs_(0) { }
  virtual ~All2All() noexcept { }
  virtual std::string Name() const noexcept override final;

  virtual void Load(const std::string& data) override final;
  virtual void Execute(float* in, float* out) const override final;
  virtual size_t inputs() const noexcept override final { return inputs_; }
  virtual size_t outputs() const noexcept override final { return neurons_.size(); }

 private:
  std::unique_ptr<float[]> weights_;
  std::unique_ptr<float[]> bias_;
  std::vector<std::unique_ptr<Neuron>> neurons_;
  size_t inputs_;
};

}  // namespace Znicz
}  // namespace Veles

#endif  // INC_ALL2ALL_H_
