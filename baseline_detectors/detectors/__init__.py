import os
import importlib
from .registry import register_detector, build_detector, get_all_registered_names
from .base import BaseDetector


def _auto_import_detectors():
    current_dir = os.path.dirname(__file__)

    for filename in os.listdir(current_dir):
        if filename.startswith("_") or not filename.endswith(
                ".py") or filename == "base.py" or filename == "registry.py":
            continue

        module_name = filename[:-3]

        try:
            importlib.import_module(f".{module_name}", package=__name__)
        except Exception as e:
            print(f"[Warning] Error occurred while automatically importing detector module '{module_name}': {e}")


_auto_import_detectors()

__all__ = ['register_detector', 'build_detector', 'get_all_registered_names', 'BaseDetector']