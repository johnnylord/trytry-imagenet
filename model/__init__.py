import sys

from .darknet import Darknet53

def get_model_cls(name):
    return getattr(sys.modules[__name__], name)
