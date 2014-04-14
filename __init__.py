# -*- coding: utf-8 -*-
"""
Created on May 21, 2013

@author: Markovtsev Vadim <v.markovtsev@samsung.com>
"""


from email.utils import parsedate_tz, mktime_tz
import os
from veles.config import root

root.common.ocl_dirs.append(os.path.join(os.path.dirname(__file__), "ocl"))


__git__ = "$Commit$"
__date__ = mktime_tz(parsedate_tz("$Date$"))
__version__ = "0.2.0"
__license__ = "Samsung Proprietary License"
__copyright__ = "© 2013 Samsung Electronics Co., Ltd."
