# -*- coding: utf-8 -*-
"""
Created on May 21, 2013

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from email.utils import parsedate_tz, mktime_tz
from sys import modules
from warnings import warn

from veles import __plugins__
from veles.config import root


__plugins__.add(modules[__name__])

root.common.compute.dirs.append("/usr/share/veles/znicz")
try:
    from .siteconfig import update
    update(root)
    del update
except ImportError:
    pass


__version__ = "0.5.0"
__license__ = "Samsung Proprietary License"
__copyright__ = "© 2013 Samsung Electronics Co., Ltd."

try:
    __git__ = "$Commit$"
    __date__ = mktime_tz(parsedate_tz("$Date$"))
except Exception as ex:
    warn("Cannot expand variables generated by Git, setting them to None")
    __git__ = None
    __date__ = None


def nothing(*args, **kwargs):
    """Does nothing (can be useful to call from unit tests).
    """
    pass
