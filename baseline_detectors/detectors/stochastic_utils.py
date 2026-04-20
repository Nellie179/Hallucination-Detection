"""
Stochastic Samples 工具函数

为需要多次采样的 detectors 提供统一的检查和生成逻辑
"""

import os
import json
import logging
import subprocess
import tempfile
from typing import List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


def load_stochastic_samples_dict(stochastic_file_path: str) -> Dict[str, List[str]]:
    """
    加载 stochastic samples 文件为字典

    Args:
        stochastic_file_path: stochastic samples jsonl 文件路径

    Returns:
        {sample_id: [samples]} 字典
    """
    samples_dict = {}

    with open(stochastic_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            sample_id = item.get("sample_id")
            samples = item.get("stochastic_samples", [])
            if sample_id:
                samples_dict[sample_id] = samples

    logger.info(f"加载了 {len(samples_dict)} 个样本的 stochastic samples")
    return samples_dict


def ensure_stochastic_samples_exist(
    metadata_file: str,
    output_file: str,
    model_name: str,
    num_samples: int = 10,
    temperature: float = 0.8,
    max_new_tokens: int = None,
    force_regenerate: bool = False
) -> Dict[str, List[str]]:
    """
    确保 stochastic samples 文件存在，如果不存在则自动生成

    Args:
        metadata_file: 输入的 metadata jsonl 文件
        output_file: 输出的 stochastic samples 文件路径
        model_name: 模型名称
        num_samples: 每个样本采样次数
        temperature: 采样温度
        max_new_tokens: 最大生成 token 数
        force_regenerate: 是否强制重新生成

    Returns:
        {sample_id: [samples]} 字典
    """
    # 检查文件是否存在
    if os.path.exists(output_file) and not force_regenerate:
        logger.info(f"检测到已存在的 stochastic samples 文件: {output_file}")
        return load_stochastic_samples_dict(output_file)

    # 文件不存在，需要生成
    logger.info(f"未找到 stochastic samples 文件，开始生成...")
    logger.info(f"  输入: {metadata_file}")
    logger.info(f"  输出: {output_file}")
    logger.info(f"  模型: {model_name}")
    logger.info(f"  采样数: {num_samples}")

    # 构建命令
    script_path = Path(__file__).parent.parent.parent / "data" / "generate_stochastic_samples.py"

    if not script_path.exists():
        raise FileNotFoundError(
            f"未找到 generate_stochastic_samples.py: {script_path}\n"
            "请确保文件存在"
        )

    cmd = [
        "python",
        str(script_path),
        "--input", metadata_file,
        "--output", output_file,
        "--model", model_name,
        "--num-samples", str(num_samples),
        "--temperature", str(temperature),
        "--trust-remote-code"
    ]

    if max_new_tokens:
        cmd.extend(["--max-new-tokens", str(max_new_tokens)])

    # 运行生成脚本
    logger.info(f"调用 generate_stochastic_samples.py...")
    logger.info(f"命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        logger.info("生成完成！")
        if result.stdout:
            logger.info(f"输出:\n{result.stdout}")

    except subprocess.CalledProcessError as e:
        logger.error(f"生成失败: {e}")
        if e.stderr:
            logger.error(f"错误信息:\n{e.stderr}")
        raise RuntimeError(f"生成 stochastic samples 失败: {e}")

    # 加载生成的文件
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"生成脚本运行完成，但输出文件不存在: {output_file}")

    return load_stochastic_samples_dict(output_file)


def infer_stochastic_file_path(metadata_file: str) -> str:
    """
    根据 metadata 文件路径推断 stochastic samples 文件路径

    例如:
        03_final_scored_metadata.jsonl -> 03_stochastic_samples.jsonl
    """
    base_dir = os.path.dirname(metadata_file)
    return os.path.join(base_dir, "03_stochastic_samples.jsonl")
