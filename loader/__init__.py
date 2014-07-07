"""
Created on Jun 7, 2014

Loader package.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


from veles.znicz.loader.base import (TRAIN, VALID, TEST, TRIAGE, CLASS_NAME,
                                     ILoader, Loader)
from veles.znicz.loader.fullbatch import (IFullBatchLoader, FullBatchLoader)
from veles.znicz.loader.image import ImageLoader