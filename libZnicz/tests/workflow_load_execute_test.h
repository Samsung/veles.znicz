/*! @file workflow_load_execute_test.h
 *  @brief New file description.
 *  @author eg.bulychev
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung HQ
 */

#ifndef WORKFLOW_LOAD_EXECUTE_TEST_H_
#define WORKFLOW_LOAD_EXECUTE_TEST_H_

#include <string>
#include <gtest/gtest.h>
#include <inc/znicz/units.h>
#include <veles/workflow_loader.h>
#include <tests/common_test.h>


namespace veles {

class WorkflowLoadExecuteTest : public CommonTest {
 public:
  WorkflowLoadExecuteTest();
  virtual ~WorkflowLoadExecuteTest() = default;

  WorkflowLoader test;
  std::string current_path;
};

}  // namespace veles
#endif  // WORKFLOW_LOAD_EXECUTE_TEST_H_
