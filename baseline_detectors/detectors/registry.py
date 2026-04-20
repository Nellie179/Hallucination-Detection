# baseline_detectors/detectors/registry.py
import inspect
from typing import Dict, Type, Any

# 全局字典，用于在内存中存储 [名称 -> 类] 的映射
_DETECTOR_REGISTRY: Dict[str, Type] = {}

def register_detector(name: str):
    """
    类装饰器：将检测器类注册到全局体系中。
    
    用法:
        @register_detector("saplma")
        class SaplmaDetector(BaseDetector):
            ...
    """
    def register_wrapper(cls: Type):
        # 【鲁棒性 1】防重复注册：防止名字冲突覆盖
        if name in _DETECTOR_REGISTRY:
            existing_cls = _DETECTOR_REGISTRY[name].__name__
            raise ValueError(
                f"[Registry Error] Detector 命名冲突！名称 '{name}' 已经被类 '{existing_cls}' 占用。"
            )

        # 【鲁棒性 2】鸭子类型检查：强制约束接口
        if not hasattr(cls, 'predict_score') or not callable(getattr(cls, 'predict_score')):
            raise TypeError(
                f"[Registry Error] 类 '{cls.__name__}' 注册失败！必须实现 'predict_score' 方法。"
            )

        _DETECTOR_REGISTRY[name] = cls
        return cls
        
    return register_wrapper

def build_detector(name: str, **kwargs) -> Any:
    """
    工厂方法：根据注册名称实例化检测器。
    """
    if name not in _DETECTOR_REGISTRY:
        available_methods = ", ".join(_DETECTOR_REGISTRY.keys())
        raise KeyError(
            f"[Registry Error] 未找到 Detector '{name}'。当前已注册的基线有: [{available_methods}]"
        )
        
    # 实例化并传入参数
    return _DETECTOR_REGISTRY[name](name=name, **kwargs)

def get_all_registered_names() -> list:
    """获取当前成功注册的所有基线名称"""
    return list(_DETECTOR_REGISTRY.keys())