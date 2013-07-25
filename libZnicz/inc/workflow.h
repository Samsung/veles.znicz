/*! @file workflow.h
 *  @brief New file description.
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef INC_WORKFLOW_H_
#define INC_WORKFLOW_H_

#include <list>
#include <string>
#include <memory>
#include <veles/unit.h>

namespace Veles {
namespace Znicz {

/** @brief VELES Neural network workflow */
class Workflow {
 public:

  /** @brief Constructs empty workflow */
  Workflow() {}
  /** @brief Constructs workflow from data in VELES format */
  explicit Workflow(const std::string& data) { Load(data); }

  void Load(const std::string& data);

  template<class InputIterator, class OutputIterator>
  void Execute(InputIterator begin, InputIterator end, OutputIterator out) const;

 private:
  size_t get_max_unit_size() const noexcept;

  std::list<std::unique_ptr<Unit>> units_;
};

}  // namespace Znicz
}  // namespace Veles

#include "workflow-inl.h"

#endif  // INC_WORKFLOW_H_
