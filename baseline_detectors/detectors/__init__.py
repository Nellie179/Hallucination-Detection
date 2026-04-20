# baseline_detectors/detectors/__init__.py
import os
import importlib
from .registry import register_detector, build_detector, get_all_registered_names
from .base import BaseDetector

def _auto_import_detectors():
    """
    【工程黑科技】动态扫描当前目录下的所有 .py 文件并自动 import。
    这确保了所有带有 @register_detector 的类在程序启动时一定会被加载到注册表中。
    """
    current_dir = os.path.dirname(__file__)
    
    for filename in os.listdir(current_dir):
        # 过滤掉隐藏文件、当前文件和非 python 文件
        if filename.startswith("_") or not filename.endswith(".py") or filename == "base.py" or filename == "registry.py":
            continue
            
        module_name = filename[:-3]  # 去掉 .py 后缀
        
        try:
            # 动态相对导入，触发该文件内的装饰器
            importlib.import_module(f".{module_name}", package=__name__)
        except Exception as e:
            print(f"[Warning] 自动导入检测器模块 '{module_name}' 时发生错误: {e}")

# 初始化包时自动执行扫描
_auto_import_detectors()

# 暴露给外部调用
__all__ = ['register_detector', 'build_detector', 'get_all_registered_names', 'BaseDetector']