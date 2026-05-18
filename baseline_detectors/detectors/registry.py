import inspect
from typing import Dict, Type, Any

_DETECTOR_REGISTRY: Dict[str, Type] = {}


def register_detector(name: str):
    def register_wrapper(cls: Type):
        if name in _DETECTOR_REGISTRY:
            existing_cls = _DETECTOR_REGISTRY[name].__name__
            raise ValueError(
                f"[Registry Error] Detector naming conflict! Name '{name}' is already occupied by class '{existing_cls}'."
            )

        if not hasattr(cls, 'predict_score') or not callable(getattr(cls, 'predict_score')):
            raise TypeError(
                f"[Registry Error] Class '{cls.__name__}' registration failed! Must implement 'predict_score' method."
            )

        _DETECTOR_REGISTRY[name] = cls
        return cls

    return register_wrapper


def build_detector(name: str, **kwargs) -> Any:
    if name not in _DETECTOR_REGISTRY:
        available_methods = ", ".join(_DETECTOR_REGISTRY.keys())
        raise KeyError(
            f"[Registry Error] Detector '{name}' not found. Currently registered baselines are: [{available_methods}]"
        )

    return _DETECTOR_REGISTRY[name](name=name, **kwargs)


def get_all_registered_names() -> list:
    return list(_DETECTOR_REGISTRY.keys())