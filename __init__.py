# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on May 21, 2013

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


from email.utils import parsedate_tz, mktime_tz
from sys import modules
from warnings import warn

from veles import __plugins__
from veles.config import root


__plugins__.add(modules[__name__])

root.common.engine.source_dirs.append("/usr/share/veles/znicz")
try:
    from .site_config import update
    update(root)
    del update
except ImportError:
    pass


__versioninfo__ = 0, 7, 0
__version__ = ".".join(map(str, __versioninfo__))
__license__ = "Samsung Proprietary License"
__copyright__ = "© 2013 Samsung Electronics Co., Ltd."

try:
    __git__ = "$Commit$"
    __date__ = mktime_tz(parsedate_tz("$Date$"))
except Exception as ex:
    warn("Cannot expand variables generated by Git, setting them to None")
    __git__ = None
    __date__ = None
