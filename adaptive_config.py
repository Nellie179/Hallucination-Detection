"""
🧠 自适应配置管理器
根据硬件能力、模型架构自动调整最优参数配置
"""

import torch
import warnings
from typing import Dict, Any, Optional


class AdaptiveConfigManager:
    """
    智能配置管理器，自动检测：
    1. GPU 架构和能力（bfloat16支持、显存等）
    2. 模型特性（架构、注意力机制等）
    3. 推荐最优的 model_kwargs 和 generation_kwargs
    """

    # GPU 架构特性数据库
    GPU_CAPABILITIES = {
        # Ampere 架构及更新 (支持 bfloat16 硬件加速)
        "ampere_plus": {
            "arch_names": ["NVIDIA A100", "NVIDIA A10", "NVIDIA A30", "NVIDIA A40",
                          "RTX 30", "RTX 40", "RTX A", "H100", "H200"],
            "preferred_dtype": "bfloat16",
            "supports_flash_attn": True
        },
        # Turing/Volta 架构 (不支持 bfloat16 硬件加速)
        "turing_volta": {
            "arch_names": ["RTX 20", "Quadro RTX", "Tesla V100", "Titan RTX", "GTX 16"],
            "preferred_dtype": "float16",
            "supports_flash_attn": True
        },
        # 更老的架构
        "legacy": {
            "arch_names": ["GTX 10", "Tesla P", "Tesla K"],
            "preferred_dtype": "float16",
            "supports_flash_attn": False
        }
    }

    # 模型特定配置（基于模型名称前缀匹配）
    MODEL_SPECIFIC_CONFIGS = {
        "Qwen": {
            "template_kwargs": {"enable_thinking": False},  # Qwen3+ 支持内部 CoT
            "trust_remote_code": True
        },
        "deepseek": {
            "template_kwargs": {"enable_thinking": True},
            "trust_remote_code": True
        },
        "Llama": {
            "trust_remote_code": False,  # Llama 无需 trust_remote_code
        },
        "Meta-Llama": {
            "trust_remote_code": False,
        },
        "mistral": {
            "trust_remote_code": False,
        },
        "gemma": {
            "trust_remote_code": False,
        },
        # 默认配置
        "_default": {
            "trust_remote_code": True  # 保守策略：默认信任
        }
    }

    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: 是否打印检测结果和推荐配置
        """
        self.verbose = verbose
        self.gpu_info = self._detect_gpu()

    def _detect_gpu(self) -> Dict[str, Any]:
        """检测 GPU 硬件能力"""
        if not torch.cuda.is_available():
            if self.verbose:
                print("[!] 警告: 未检测到 CUDA GPU，将使用 CPU 模式")
            return {
                "available": False,
                "name": "CPU",
                "arch_category": "cpu",
                "preferred_dtype": "float32",
                "supports_flash_attn": False,
                "compute_capability": (0, 0)
            }

        gpu_name = torch.cuda.get_device_name(0)
        compute_cap = torch.cuda.get_device_capability(0)

        # 判断架构类别
        arch_category = "legacy"
        for cat, info in self.GPU_CAPABILITIES.items():
            if any(keyword in gpu_name for keyword in info["arch_names"]):
                arch_category = cat
                break

        gpu_info = {
            "available": True,
            "name": gpu_name,
            "arch_category": arch_category,
            "preferred_dtype": self.GPU_CAPABILITIES.get(arch_category, {}).get("preferred_dtype", "float16"),
            "supports_flash_attn": compute_cap >= (7, 5),  # SM 7.5+ 支持
            "compute_capability": compute_cap,
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
        }

        if self.verbose:
            print("=" * 60)
            print("🔍 GPU 硬件检测结果")
            print("=" * 60)
            print(f"GPU 名称: {gpu_info['name']}")
            print(f"架构类别: {gpu_info['arch_category']}")
            print(f"计算能力: SM {compute_cap[0]}.{compute_cap[1]}")
            print(f"推荐精度: {gpu_info['preferred_dtype']}")
            print(f"显存容量: {gpu_info['total_memory_gb']:.1f} GB")
            print(f"Flash Attention 支持: {'✓' if gpu_info['supports_flash_attn'] else '✗'}")
            print("=" * 60 + "\n")

        return gpu_info

    def _get_model_family(self, model_name: str) -> str:
        """从模型名称提取模型家族"""
        for family in self.MODEL_SPECIFIC_CONFIGS.keys():
            if family.lower() in model_name.lower():
                return family
        return "_default"

    def get_optimal_config(
        self,
        model_name: str,
        user_model_kwargs: Optional[Dict[str, Any]] = None,
        user_generation_kwargs: Optional[Dict[str, Any]] = None,
        user_template_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        为给定模型和硬件生成最优配置

        Args:
            model_name: 模型名称 (如 "Qwen/Qwen3-8B")
            user_model_kwargs: 用户自定义的 model_kwargs（会覆盖自动配置）
            user_generation_kwargs: 用户自定义的 generation_kwargs
            user_template_kwargs: 用户自定义的 template_kwargs

        Returns:
            包含 model_kwargs, generation_kwargs, template_kwargs 的字典
        """
        model_family = self._get_model_family(model_name)
        model_config = self.MODEL_SPECIFIC_CONFIGS.get(
            model_family,
            self.MODEL_SPECIFIC_CONFIGS["_default"]
        )

        # ========== 1. 构建 model_kwargs ==========
        optimal_model_kwargs = {
            "device_map": "auto",  # 让 Transformers 自动分配设备
        }

        # 添加信任代码设置
        if "trust_remote_code" in model_config:
            optimal_model_kwargs["trust_remote_code"] = model_config["trust_remote_code"]

        # 根据 GPU 能力设置精度
        if self.gpu_info["available"]:
            preferred_dtype_str = self.gpu_info["preferred_dtype"]
            # 转换为 torch dtype 对象（使用 dtype 而非 torch_dtype 以避免 deprecation 警告）
            if preferred_dtype_str == "bfloat16":
                optimal_model_kwargs["dtype"] = torch.bfloat16
            elif preferred_dtype_str == "float16":
                optimal_model_kwargs["dtype"] = torch.float16
            else:
                optimal_model_kwargs["dtype"] = torch.float32

            # 自适应选择注意力实现
            if self.gpu_info["supports_flash_attn"]:
                # 尝试使用 Flash Attention
                try:
                    import flash_attn
                    # 验证 flash_attn 是否可用
                    try:
                        from flash_attn import flash_attn_func
                        # 成功导入，可以使用 flash_attention_2
                        optimal_model_kwargs["attn_implementation"] = "flash_attention_2"
                        if self.verbose:
                            try:
                                fa_version = flash_attn.__version__
                            except:
                                try:
                                    from importlib.metadata import version
                                    fa_version = version('flash-attn')
                                except:
                                    fa_version = "unknown"
                            print(f"[✓] Flash Attention 可用 (版本 {fa_version})，将使用 flash_attention_2")
                    except ImportError:
                        # flash_attn 包存在但不完整，降级到 SDPA
                        optimal_model_kwargs["attn_implementation"] = "sdpa"
                        if self.verbose:
                            print("[!] Flash Attention 安装不完整，降级到 SDPA (性能仍然很好)")
                except ImportError:
                    # 未安装 flash-attn，使用 SDPA
                    optimal_model_kwargs["attn_implementation"] = "sdpa"
                    if self.verbose:
                        print("[!] Flash Attention 未安装，使用 SDPA")
                        print("    如需更好性能，可安装: pip install flash-attn --no-build-isolation")
            else:
                # 旧 GPU 使用默认的 eager 模式
                optimal_model_kwargs["attn_implementation"] = "eager"

        # 用户自定义配置覆盖（高优先级）
        if user_model_kwargs:
            optimal_model_kwargs.update(user_model_kwargs)

        # ========== 2. 构建 generation_kwargs ==========
        optimal_generation_kwargs = {
            "do_sample": False,  # 默认贪婪解码，确保可复现
        }

        if user_generation_kwargs:
            optimal_generation_kwargs.update(user_generation_kwargs)

            # 智能清理：如果 do_sample=False，移除采样参数
            if not optimal_generation_kwargs.get("do_sample", False):
                for key in ["temperature", "top_p", "top_k"]:
                    optimal_generation_kwargs.pop(key, None)

        # ========== 3. 构建 template_kwargs ==========
        optimal_template_kwargs = model_config.get("template_kwargs", {}).copy()

        if user_template_kwargs:
            optimal_template_kwargs.update(user_template_kwargs)

        # ========== 打印最终配置 ==========
        if self.verbose:
            print("=" * 60)
            print("⚙️  自适应配置生成结果")
            print("=" * 60)
            print(f"目标模型: {model_name}")
            print(f"模型家族: {model_family}")
            print(f"\n推荐配置:")
            print(f"  model_kwargs: {optimal_model_kwargs}")
            print(f"  generation_kwargs: {optimal_generation_kwargs}")
            print(f"  template_kwargs: {optimal_template_kwargs}")
            print("=" * 60 + "\n")

        return {
            "model_kwargs": optimal_model_kwargs,
            "generation_kwargs": optimal_generation_kwargs,
            "template_kwargs": optimal_template_kwargs
        }

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        验证配置合法性，给出警告

        Args:
            config: 包含 model_kwargs, generation_kwargs 等的配置字典
        """
        model_kwargs = config.get("model_kwargs", {})
        generation_kwargs = config.get("generation_kwargs", {})

        # 检查 bfloat16 在不支持的 GPU 上使用
        if (model_kwargs.get("torch_dtype") == "bfloat16" and
            self.gpu_info["arch_category"] not in ["ampere_plus"]):
            warnings.warn(
                f"⚠️  您在 {self.gpu_info['name']} 上使用 bfloat16，但该 GPU 不支持硬件加速。"
                f"推荐改用 float16 以获得更好性能。",
                UserWarning
            )

        # 检查 flex_attention 未编译
        if model_kwargs.get("attn_implementation") == "flex_attention":
            warnings.warn(
                "⚠️  flex_attention 需要 torch.compile() 才能高效运行。"
                "推荐改用 'sdpa' 或删除该参数使用默认值。",
                UserWarning
            )

        # 检查无效的采样参数
        if not generation_kwargs.get("do_sample", False):
            invalid_params = [k for k in ["temperature", "top_p", "top_k"]
                            if k in generation_kwargs]
            if invalid_params:
                warnings.warn(
                    f"⚠️  do_sample=False 时，以下参数无效: {invalid_params}",
                    UserWarning
                )


# ==========================================
# 便捷函数：一键获取最优配置
# ==========================================
def get_adaptive_config(
    model_name: str,
    user_model_kwargs: Optional[Dict[str, Any]] = None,
    user_generation_kwargs: Optional[Dict[str, Any]] = None,
    user_template_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    一键获取自适应配置（推荐使用此函数）

    示例:
        config = get_adaptive_config("Qwen/Qwen3-8B")
        extractor = HiddenStateExtractor(
            model_name="Qwen/Qwen3-8B",
            model_kwargs=config["model_kwargs"]
        )
    """
    manager = AdaptiveConfigManager(verbose=verbose)
    return manager.get_optimal_config(
        model_name=model_name,
        user_model_kwargs=user_model_kwargs,
        user_generation_kwargs=user_generation_kwargs,
        user_template_kwargs=user_template_kwargs
    )


# ==========================================
# 单元测试
# ==========================================
if __name__ == "__main__":
    print("\n" + "🧪 开始自适应配置模块测试".center(60, "=") + "\n")

    # 测试不同模型
    test_models = [
        "Qwen/Qwen3-8B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "mistralai/Mistral-7B-v0.1",
        "deepseek-ai/deepseek-coder-6.7b-instruct"
    ]

    for model in test_models:
        print(f"\n{'Testing: ' + model:=^60}\n")
        config = get_adaptive_config(model)

    # 测试用户覆盖
    print(f"\n{'测试用户自定义覆盖':=^60}\n")
    custom_config = get_adaptive_config(
        "Qwen/Qwen3-8B",
        user_model_kwargs={"attn_implementation": "eager"},
        user_generation_kwargs={"do_sample": True, "temperature": 0.7}
    )

    print("\n" + "✅ 测试完成".center(60, "=") + "\n")
