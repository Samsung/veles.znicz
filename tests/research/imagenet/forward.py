"""
Created on Jul 9, 2014

Copyright (c) 2014, Samsung Electronics, Co., Ltd.
"""


from veles.znicz.loader import Loader


class ForwardStage1(Loader):
    """
    Imagenet loader for first processing stage.
    """

    def __init__(self, workflow, **kwargs):
        super(ForwardStage1, self).__init__(workflow, **kwargs)
