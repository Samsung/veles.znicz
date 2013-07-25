/*! @file neuron.h
 *  @brief VELES neuron, defined by its activation function
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef INC_NEURON_H_
#define INC_NEURON_H_

#include <string>

namespace Veles {
namespace Znicz {

/** @brief VELES neuron */
class Neuron
{
 public:
  virtual ~Neuron() noexcept { }
  virtual std::string Name() const noexcept = 0;

  virtual void Load(const std::string& data) = 0;
  virtual float Execute(float data) const = 0;
};

}  // namespace Znicz
}  // namespace Veles

#endif  // INC_NEURON_H_
