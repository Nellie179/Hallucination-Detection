#!/usr/bin/env python3
"""
提取Q+A拼接后的完整Hidden States

用途:
    某些baseline检测器(SAPLMA, MIND, ICR Probe等)需要Question和Answer拼接后
    一起输入LLM得到的hidden states,而不是只有Answer部分的hidden states。

    本脚本用于从原始数据中重新提取这些完整的hidden states。

输入:
    - 原始数据文件(包含question和answer)
    - 模型checkpoint路径

输出:
    - HDF5文件,包含Q+A的完整hidden states

结构:
    sample_id/
        full_hidden_states/  # Q+A拼接的hidden states
            token_0/
                layer_16/
                layer_17/
                ...
            token_1/
            ...
        question_length: int  # question的token长度
        answer_start_idx: int # answer开始的token索引

使用方法:
    python extract_full_hidden_states.py \\
        --input_file data/samples.json \\
        --output_file data/full_hidden_states.h5 \\
        --model_name meta-llama/Llama-2-7b-hf \\
        --batch_size 8 \\
        --layers 16,17,18,19,20  # 或 "all" 提取所有层
"""

import argparse
import json
import h5py
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HiddenStatesExtractor:
    """提取Q+A拼接的完整hidden states"""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        layers: List[int] = None,
        dtype: torch.dtype = torch.float16
    ):
        """
        Args:
            model_name: 模型名称或路径
            device: 设备
            layers: 要提取的层索引列表,None表示所有层
            dtype: 模型数据类型
        """
        logger.info(f"加载模型: {model_name}")
        logger.info(f"设备: {device}")

        self.device = device
        self.dtype = dtype
        self.layers = layers

        # 加载tokenizer和model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()

        # 获取层数
        self.num_layers = self.model.config.num_hidden_layers
        logger.info(f"模型层数: {self.num_layers}")

        if self.layers is None:
            self.layers = list(range(self.num_layers))
        logger.info(f"提取层: {self.layers}")

    def extract_hidden_states(
        self,
        question: str,
        answer: str
    ) -> Tuple[List[np.ndarray], int]:
        """
        提取Q+A的hidden states

        Args:
            question: 问题文本
            answer: 答案文本

        Returns:
            (hidden_states_list, question_length)
            - hidden_states_list: 每个layer的hidden states, shape [seq_len, hidden_dim]
            - question_length: question的token数量
        """
        # 拼接Q+A
        full_text = question + " " + answer

        # Tokenize
        question_tokens = self.tokenizer(question, add_special_tokens=True)
        full_tokens = self.tokenizer(full_text, add_special_tokens=True, return_tensors="pt")

        question_length = len(question_tokens['input_ids'])

        # 移到设备
        input_ids = full_tokens['input_ids'].to(self.device)
        attention_mask = full_tokens['attention_mask'].to(self.device)

        # 提取hidden states
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

        # outputs.hidden_states是一个tuple,包含每一层的输出
        # 每个元素shape: [batch_size=1, seq_len, hidden_dim]
        hidden_states_by_layer = {}

        for layer_idx in self.layers:
            # +1因为第0个是embedding layer
            hidden = outputs.hidden_states[layer_idx + 1]  # [1, seq_len, hidden_dim]
            hidden = hidden.squeeze(0).cpu().numpy()  # [seq_len, hidden_dim]

            # 转换为float16节省空间
            hidden = hidden.astype(np.float16)

            hidden_states_by_layer[layer_idx] = hidden

        return hidden_states_by_layer, question_length

    def extract_batch(
        self,
        samples: List[Dict],
        output_file: str
    ):
        """
        批量提取并保存到HDF5

        Args:
            samples: 样本列表,每个样本包含 {sample_id, question, answer}
            output_file: 输出HDF5文件路径
        """
        logger.info(f"开始提取 {len(samples)} 个样本的hidden states")
        logger.info(f"输出文件: {output_file}")

        with h5py.File(output_file, 'w') as f:
            for sample in tqdm(samples, desc="提取hidden states"):
                try:
                    sample_id = sample['sample_id']
                    question = sample['question']
                    answer = sample['answer']

                    # 提取hidden states
                    hidden_states_by_layer, question_length = self.extract_hidden_states(
                        question, answer
                    )

                    # 保存到HDF5
                    sample_group = f.create_group(sample_id)

                    # 保存metadata
                    sample_group.attrs['question_length'] = question_length
                    sample_group.attrs['answer_start_idx'] = question_length
                    sample_group.attrs['question'] = question
                    sample_group.attrs['answer'] = answer

                    # 保存hidden states
                    # 结构: sample_id/token_i/layer_j
                    seq_len = list(hidden_states_by_layer.values())[0].shape[0]

                    for token_idx in range(seq_len):
                        token_group = sample_group.create_group(f"token_{token_idx}")

                        for layer_idx, hidden_states in hidden_states_by_layer.items():
                            # hidden_states: [seq_len, hidden_dim]
                            token_hidden = hidden_states[token_idx, :]  # [hidden_dim]
                            token_group.create_dataset(
                                f"layer_{layer_idx}",
                                data=token_hidden,
                                compression="gzip",
                                compression_opts=4
                            )

                except Exception as e:
                    logger.error(f"样本 {sample.get('sample_id', 'unknown')} 提取失败: {e}")
                    continue

        logger.info(f"提取完成! 保存到: {output_file}")


def load_samples_from_json(input_file: str) -> List[Dict]:
    """
    从JSON文件加载样本

    Args:
        input_file: JSON文件路径

    Returns:
        样本列表
    """
    logger.info(f"加载样本: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 假设格式是列表或字典
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        samples = list(data.values())
    else:
        raise ValueError(f"不支持的数据格式: {type(data)}")

    logger.info(f"加载了 {len(samples)} 个样本")
    return samples


def main():
    parser = argparse.ArgumentParser(description='提取Q+A的完整hidden states')

    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='输入JSON文件路径'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='输出HDF5文件路径'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='模型名称或路径'
    )
    parser.add_argument(
        '--layers',
        type=str,
        default='all',
        help='要提取的层,用逗号分隔(如 "16,17,18"),或 "all" 提取所有层'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='设备: cuda 或 cpu'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='float16',
        choices=['float16', 'float32', 'bfloat16'],
        help='模型数据类型'
    )

    args = parser.parse_args()

    # 解析layers参数
    if args.layers.lower() == 'all':
        layers = None
    else:
        layers = [int(x.strip()) for x in args.layers.split(',')]

    # 解析dtype
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]

    # 加载样本
    samples = load_samples_from_json(args.input_file)

    # 创建extractor
    extractor = HiddenStatesExtractor(
        model_name=args.model_name,
        device=args.device,
        layers=layers,
        dtype=dtype
    )

    # 提取并保存
    extractor.extract_batch(samples, args.output_file)

    logger.info("全部完成!")


if __name__ == "__main__":
    main()
