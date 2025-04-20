# models/__init__.py

from .detr import build


def build_model(args):
    return build(args)
