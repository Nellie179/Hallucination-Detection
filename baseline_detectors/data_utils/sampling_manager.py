import os
import json
import subprocess
import hashlib
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
logger = logging.getLogger("SamplingManager")

class SamplingManager:
    def __init__(self, config):
        self.config = config
        self.original_metadata_path = config.METADATA_JSONL
        self.sampling_cache_dir = getattr(config, 'SAMPLING_CACHE_DIR',
                                         os.path.join(os.path.dirname(self.original_metadata_path), "sampling_cache"))
        self.model_name = config.DEFAULT_MODEL

        os.makedirs(self.sampling_cache_dir, exist_ok=True)

        self.generator_path = self._find_generator_path()

        logger.info("Initialization complete")
        logger.info(f"  Original metadata: {self.original_metadata_path}")
        logger.info(f"  Cache directory: {self.sampling_cache_dir}")
        logger.info(f"  Generator path: {self.generator_path}")

    def _find_generator_path(self) -> str:
        current_dir = os.path.dirname(os.path.dirname(__file__))
        possible_paths = [
            os.path.join(current_dir, "../data/generate_stochastic_samples.py"),
            os.path.join(current_dir, "../../data/generate_stochastic_samples.py"),
            "./data/generate_stochastic_samples.py",
        ]

        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path

        raise FileNotFoundError(
            "Cannot find sampling generator script generate_stochastic_samples.py\n"
            f"Attempted paths: {possible_paths}"
        )

    def ensure_samples_available(
            self,
            num_samples: int = 10,
            temperature: float = 0.8,
            force_regenerate: bool = False
    ) -> str:
        cached_path = self._get_cached_path(num_samples, temperature)

        if not force_regenerate and self._validate_samples(cached_path, num_samples):
            logger.info(f"✓ Using cached sampled data: {cached_path}")
            return cached_path

        logger.info(f"Starting to generate sampled data: num_samples={num_samples}, temperature={temperature}")
        return self._generate_samples(num_samples, temperature, cached_path)

    def _get_cached_path(self, num_samples: int, temperature: float) -> str:
        cached_filename = f"samples_n{num_samples}_t{temperature:.2f}.jsonl"
        cached_path = os.path.join(self.sampling_cache_dir, cached_filename)
        return cached_path

    def _validate_samples(self, path: str, expected_num: int) -> bool:
        if not os.path.exists(path):
            logger.debug(f"Cache file does not exist: {path}")
            return False

        try:
            if os.path.getsize(path) == 0:
                logger.warning(f"Cache file is empty: {path}")
                return False

            valid_count = 0
            with open(path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 5:
                        break

                    if not line.strip():
                        continue

                    item = json.loads(line)

                    if "stochastic_samples" not in item:
                        logger.warning(f"Missing stochastic_samples field: {path} (line {i+1})")
                        return False

                    samples = item["stochastic_samples"]

                    if len(samples) < expected_num:
                        logger.warning(
                            f"Insufficient samples: expected {expected_num}, actual {len(samples)} "
                            f"({path}, line {i+1})"
                        )
                        return False

                    valid_count += 1

            logger.debug(f"Validation passed: {path} (checked {valid_count} lines)")
            return True

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {path} - {e}")
            return False
        except Exception as e:
            logger.error(f"Validation failed: {path} - {e}")
            return False

    def _generate_samples(
            self,
            num_samples: int,
            temperature: float,
            output_path: str
    ) -> str:
        cmd = [
            "python3",
            self.generator_path,
            "--input", self.original_metadata_path,
            "--output", output_path,
            "--model", self.model_name,
            "--num-samples", str(num_samples),
            "--temperature", str(temperature),
            "--trust-remote-code",
            "--resume"
        ]

        logger.info(f"Executing command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                timeout=7200,
                capture_output=True,
                text=True
            )

            logger.info("✓ Sampling generation complete")
            logger.debug(f"Output:\n{result.stdout}")

            if not self._validate_samples(output_path, num_samples):
                raise RuntimeError(f"Generated file validation failed: {output_path}")

            return output_path

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Sampling generation timeout (2 hours)!\n"
                f"Command: {' '.join(cmd)}"
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Sampling generation failed!\n"
                f"Command: {' '.join(cmd)}\n"
                f"Error: {e.stderr}"
            )
        except Exception as e:
            raise RuntimeError(f"Error occurred during sampling generation: {e}")

    def get_cache_info(self) -> dict:
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
        if not confirm:
            logger.warning("Clearing cache requires confirm=True")
            return

        if not os.path.exists(self.sampling_cache_dir):
            logger.info("Cache directory does not exist, no need to clear")
            return

        import shutil
        shutil.rmtree(self.sampling_cache_dir)
        os.makedirs(self.sampling_cache_dir, exist_ok=True)

        logger.info(f"✓ Cache directory cleared: {self.sampling_cache_dir}")


if __name__ == "__main__":
    print("=" * 70)
    print("SamplingManager Unit Test")
    print("=" * 70)

    class MockConfig:
        METADATA_JSONL = "./experiments/Qwen/Qwen3-8B/coqa_5000samples/03_final_scored_metadata.jsonl"
        DEFAULT_MODEL = "Qwen/Qwen3-8B"
        SAMPLING_CACHE_DIR = "./experiments/Qwen/Qwen3-8B/coqa_5000samples/sampling_cache"

    try:
        config = MockConfig()
        manager = SamplingManager(config)

        print("\n[*] Cache Info:")
        cache_info = manager.get_cache_info()
        print(f"    Cache directory: {cache_info['cache_dir']}")
        print(f"    Total files: {cache_info['total_files']}")
        print(f"    Total size: {cache_info['total_size_mb']:.2f} MB")

        print("\n[*] Testing cache path generation:")
        path1 = manager._get_cached_path(10, 0.8)
        print(f"    num_samples=10, temp=0.8 -> {os.path.basename(path1)}")

        path2 = manager._get_cached_path(20, 0.9)
        print(f"    num_samples=20, temp=0.9 -> {os.path.basename(path2)}")

        print("\n✅ Test complete")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()