/*! @file workflow_load_execute_test.cc
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
#include <unistd.h>
#include <tests/workflow_load_execute_test.h>

using std::string;

namespace veles {

WorkflowLoadExecuteTest::WorkflowLoadExecuteTest() {
  current_path = "";
  char  currentPath[FILENAME_MAX];

  if (!getcwd(currentPath, sizeof(currentPath))) {
    ERR("Can't locate current directory");
  } else {
    current_path = currentPath;
    DBG("current_path: %s", current_path.c_str());
  }
  string temp_path1 = current_path + "/workflow_files/";
  string temp_path2 = current_path + "/tests/workflow_files/";
  string temp_path3 = current_path + "/../workflow_files/";
  // Check existence of archive
  if (access(temp_path1.c_str(), 0) != -1) {
    current_path = temp_path1;  // "/workflow_desc_files/"
  } else if (access(temp_path2.c_str(), 0) != -1) {
    current_path = temp_path2;  // "/tests/workflow_desc_files/"
  } else if (access(temp_path3.c_str(), 0) != -1) {
    current_path = temp_path3;  // "/../workflow_desc_files/"
  } else {
    current_path = "";  // Error
  }
  DBG("Path to workflow files: %s", current_path.c_str());

}

TEST_F(WorkflowLoadExecuteTest, LoadExecute) {
  auto fn = current_path + "channels_workflow.tar.gz";
  ASSERT_NO_THROW(test.Load(fn));
  Workflow result;
  ASSERT_NO_THROW(result = test.GetWorkflow());

  DBG("Length of input float array = %zu", result.InputCount());
  DBG("Length of output float array = %zu", result.OutputCount());

  auto input = CreateFloatArray(result.InputCount());
  auto output = CreateFloatArray(result.OutputCount());

  ASSERT_NO_THROW(result.Execute(input.get(),
                                 input.get() + result.InputCount() - 1,
                                 output.get()));
  DBG("Execution completed.");
}

}  // namespace veles

#include "tests/google/src/gtest_main.cc"
