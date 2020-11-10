import sys

from .imagenet import ImageNetAgent


def get_agent_cls(name):
    return getattr(sys.modules[__name__], name)
