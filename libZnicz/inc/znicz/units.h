/*! @file units.h
 *  @brief The forward declarations of all Znicz units.
 *  @author Vadim Markovtsev <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright © 2013 Samsung R&D Institute Russia
 *
 *  @section License
 *  Licensed to the Apache Software Foundation (ASF) under one
 *  or more contributor license agreements.  See the NOTICE file
 *  distributed with this work for additional information
 *  regarding copyright ownership.  The ASF licenses this file
 *  to you under the Apache License, Version 2.0 (the
 *  "License"); you may not use this file except in compliance
 *  with the License.  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an
 *  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 *  KIND, either express or implied.  See the License for the
 *  specific language governing permissions and limitations
 *  under the License.
 */

#ifndef INC_ZNICZ_UNITS_H_
#define INC_ZNICZ_UNITS_H_

#include <veles/unit_factory.h>

#if __GNUC__ >= 4
#pragma GCC visibility push(default)
#endif

namespace veles {
namespace znicz {

class All2AllTanh;
class All2AllSoftmax;
class All2AllLinear;

DECLARE_UNIT(All2AllLinear);
DECLARE_UNIT(All2AllSoftmax);
DECLARE_UNIT(All2AllTanh);

}  // namespace znicz
}  // namespace veles

#if __GNUC__ >= 4
#pragma GCC visibility pop
#endif

#endif  // INC_ZNICZ_UNITS_H_
