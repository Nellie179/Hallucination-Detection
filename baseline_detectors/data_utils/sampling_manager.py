# baseline_detectors/data_utils/sampling_manager.py
"""
SamplingManager - 采样数据管理器

职责：
    1. 检查 metadata 是否包含 stochastic_samples
    2. 如果缺失，自动调用生成器创建
    3. 缓存已生成的数据路径
    4. 提供统一的采样数据访问接口

使用方式：
    manager = SamplingManager(config)
    sampled_path = manager.ensure_samples_available(num_samples=10, temperature=0.8)
"""

import os
import json
import subprocess
import hashlib
import logging
from typing import Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
logger = logging.getLogger("SamplingManager")


class SamplingManager:
    """
    采样数据管理器
    """

    def __init__(self, config):
        """
        Args:
            config: 全局配置对象（config.py）
        """
        self.config = config
        self.original_metadata_path = config.METADATA_JSONL
        self.sampling_cache_dir = getattr(config, 'SAMPLING_CACHE_DIR',
                                         os.path.join(os.path.dirname(self.original_metadata_path), "sampling_cache"))
        self.model_name = config.DEFAULT_MODEL

        # 创建缓存目录
        os.makedirs(self.sampling_cache_dir, exist_ok=True)

        # 采样生成器路径
        self.generator_path = self._find_generator_path()

        logger.info(f"初始化完成")
        logger.info(f"  原始 metadata: {self.original_metadata_path}")
        logger.info(f"  缓存目录: {self.sampling_cache_dir}")
        logger.info(f"  生成器路径: {self.generator_path}")

    def _find_generator_path(self) -> str:
        """
        查找采样生成器脚本路径
        """
        # 可能的路径（相对于当前文件）
        current_dir = os.path.dirname(os.path.dirname(__file__))
        possible_paths = [
            os.path.join(current_dir, "../data/generate_stochastic_samples.py"),
            os.path.join(current_dir, "../../data/generate_stochastic_samples.py"),
            "/home/zfang1/Data/Lxy/Benchmark/data/generate_stochastic_samples.py",
        ]

        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path

        raise FileNotFoundError(
            "找不到采样生成器脚本 generate_stochastic_samples.py\n"
            f"已尝试路径: {possible_paths}"
        )

    def ensure_samples_available(
            self,
            num_samples: int = 10,
            temperature: float = 0.8,
            force_regenerate: bool = False
    ) -> str:
        """
        确保采样数据可用

        Args:
            num_samples: 每个样本的采样次数
            temperature: 采样温度
            force_regenerate: 是否强制重新生成（忽略缓存）

        Returns:
            包含采样数据的 metadata 文件路径
        """
        # 生成缓存文件名
        cached_path = self._get_cached_path(num_samples, temperature)

        # 检查缓存是否有效
        if not force_regenerate and self._validate_samples(cached_path, num_samples):
            logger.info(f"✓ 使用缓存的采样数据: {cached_path}")
            return cached_path

        # 生成新采样
        logger.info(f"开始生成采样数据: num_samples={num_samples}, temperature={temperature}")
        return self._generate_samples(num_samples, temperature, cached_path)

    def _get_cached_path(self, num_samples: int, temperature: float) -> str:
        """
        生成缓存文件路径（独立的采样文件）

        格式: sampling_cache/samples_n{num}_t{temp}.jsonl
        """
        # 简化文件名，只包含采样配置信息
        cached_filename = f"samples_n{num_samples}_t{temperature:.2f}.jsonl"
        cached_path = os.path.join(self.sampling_cache_dir, cached_filename)
        return cached_path

    def _validate_samples(self, path: str, expected_num: int) -> bool:
        """
        验证采样数据的完整性和有效性

        Args:
            path: 待验证的文件路径
            expected_num: 期望的采样数量

        Returns:
            是否有效
        """
        if not os.path.exists(path):
            logger.debug(f"缓存文件不存在: {path}")
            return False

        try:
            # 检查文件是否为空
            if os.path.getsize(path) == 0:
                logger.warning(f"缓存文件为空: {path}")
                return False

            # 抽样检查前 5 行
            valid_count = 0
            with open(path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 5:  # 只检查前 5 行
                        break

                    if not line.strip():
                        continue

                    item = json.loads(line)

                    # 检查是否有 stochastic_samples 字段
                    if "stochastic_samples" not in item:
                        logger.warning(f"缺少 stochastic_samples 字段: {path} (line {i+1})")
                        return False

                    samples = item["stochastic_samples"]

                    # 检查采样数量
                    if len(samples) < expected_num:
                        logger.warning(
                            f"采样数不足: 期望 {expected_num}，实际 {len(samples)} "
                            f"({path}, line {i+1})"
                        )
                        return False

                    valid_count += 1

            logger.debug(f"验证通过: {path} (检查了 {valid_count} 行)")
            return True

        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {path} - {e}")
            return False
        except Exception as e:
            logger.error(f"验证失败: {path} - {e}")
            return False

    def _generate_samples(
            self,
            num_samples: int,
            temperature: float,
            output_path: str
    ) -> str:
        """
        调用采样生成器创建新的采样数据

        Args:
            num_samples: 采样次数
            temperature: 采样温度
            output_path: 输出文件路径

        Returns:
            生成的文件路径
        """
        # 构建命令
        cmd = [
            "python3",
            self.generator_path,
            "--input", self.original_metadata_path,
            "--output", output_path,
            "--model", self.model_name,
            "--num-samples", str(num_samples),
            "--temperature", str(temperature),
            "--trust-remote-code",
            "--resume"  # 支持断点续传
        ]

        logger.info(f"执行命令: {' '.join(cmd)}")

        try:
            # 执行生成器（设置 2 小时超时）
            result = subprocess.run(
                cmd,
                check=True,
                timeout=7200,  # 2 小时
                capture_output=True,
                text=True
            )

            logger.info("✓ 采样生成完成")
            logger.debug(f"输出:\n{result.stdout}")

            # 验证生成的文件
            if not self._validate_samples(output_path, num_samples):
                raise RuntimeError(f"生成的文件验证失败: {output_path}")

            return output_path

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"采样生成超时（2小时）！\n"
                f"命令: {' '.join(cmd)}"
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"采样生成失败！\n"
                f"命令: {' '.join(cmd)}\n"
                f"错误: {e.stderr}"
            )
        except Exception as e:
            raise RuntimeError(f"采样生成过程中发生错误: {e}")

    def get_cache_info(self) -> dict:
        """
        获取缓存信息（用于调试和监控）

        Returns:
            包含缓存统计的字典
        """
        if not os.path.exists(self.sampling_cache_dir):
            return {"cache_dir": self.sampling_cache_dir, "cached_files": []}

        cached_files = []
        total_size = 0

        for filename in os.listdir(self.sampling_cache_dir):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(self.sampling_cache_dir, filename)
                size = os.path.getsize(filepath)
                total_size += size

                cached_files.append({
                    "filename": filename,
                    "size_mb": size / 1024 / 1024,
                    "path": filepath
                })

        return {
            "cache_dir": self.sampling_cache_dir,
            "cached_files": cached_files,
            "total_files": len(cached_files),
            "total_size_mb": total_size / 1024 / 1024
        }

    def clear_cache(self, confirm: bool = False):
        """
        清空缓存目录（谨慎使用）

        Args:
            confirm: 必须设为 True 才会执行
        """
        if not confirm:
            logger.warning("清空缓存需要设置 confirm=True")
            return

        if not os.path.exists(self.sampling_cache_dir):
            logger.info("缓存目录不存在，无需清空")
            return

        import shutil
        shutil.rmtree(self.sampling_cache_dir)
        os.makedirs(self.sampling_cache_dir, exist_ok=True)

        logger.info(f"✓ 已清空缓存目录: {self.sampling_cache_dir}")


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print("SamplingManager 单元测试")
    print("=" * 70)

    # 模拟 config 对象
    class MockConfig:
        METADATA_JSONL = "/home/zfang1/Data/Lxy/Benchmark/experiments/Qwen/Qwen3-8B/coqa_5000samples/03_final_scored_metadata.jsonl"
        DEFAULT_MODEL = "Qwen/Qwen3-8B"
        SAMPLING_CACHE_DIR = "/home/zfang1/Data/Lxy/Benchmark/experiments/Qwen/Qwen3-8B/coqa_5000samples/sampling_cache"

    try:
        config = MockConfig()
        manager = SamplingManager(config)

        # 查看缓存信息
        print("\n[*] 缓存信息:")
        cache_info = manager.get_cache_info()
        print(f"    缓存目录: {cache_info['cache_dir']}")
        print(f"    缓存文件数: {cache_info['total_files']}")
        print(f"    总大小: {cache_info['total_size_mb']:.2f} MB")

        # 测试缓存路径生成
        print("\n[*] 测试缓存路径生成:")
        path1 = manager._get_cached_path(10, 0.8)
        print(f"    num_samples=10, temp=0.8 -> {os.path.basename(path1)}")

        path2 = manager._get_cached_path(20, 0.9)
        print(f"    num_samples=20, temp=0.9 -> {os.path.basename(path2)}")

        print("\n✅ 测试完成")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
