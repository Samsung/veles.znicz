/*! @file all2all.h
 *  @brief New file description.
 *  @author Markovtsev Vadim <v.markovtsev@samsung.com>
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

#include <veles/unit.h>

namespace Veles {
namespace Znicz {

/// @brief "All to all" neural network layer.
class All2All : public Unit {
 public:
  virtual std::string Name() const noexcept override final;
};

}  // namespace Znicz
}  // namespace Veles

#endif  // SRC_ALL2ALL_H_
