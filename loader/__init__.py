"""
Created on Jun 7, 2014

Loader package.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


from veles.znicz.loader.base import (TRAIN, VALID, TEST, TRIAGE, CLASS_NAME,
                                     ILoader, Loader, LoaderMSE)
from veles.znicz.loader.fullbatch import (IFullBatchLoader, FullBatchLoader,
                                          FullBatchLoaderMSE)
from veles.znicz.loader.image import (
    IFileImageLoader, FullBatchFileImageLoader, FullBatchFileImageLoaderMSE,
    FullBatchAutoLabelFileImageLoader)

from veles.znicz.loader.pickles import PicklesImageFullBatchLoader
