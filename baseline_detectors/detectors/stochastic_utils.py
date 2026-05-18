import os
import json
import logging
import subprocess
import tempfile
from typing import List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


def load_stochastic_samples_dict(stochastic_file_path: str) -> Dict[str, List[str]]:
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

    logger.info(f"Loaded stochastic samples for {len(samples_dict)} items")
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
    if os.path.exists(output_file) and not force_regenerate:
        logger.info(f"Detected existing stochastic samples file: {output_file}")
        return load_stochastic_samples_dict(output_file)

    logger.info("Stochastic samples file not found, initializing generator...")
    logger.info(f"  Input: {metadata_file}")
    logger.info(f"  Output: {output_file}")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Sample count: {num_samples}")

    script_path = Path(__file__).parent.parent.parent / "data" / "generate_stochastic_samples.py"

    if not script_path.exists():
        raise FileNotFoundError(
            f"Could not find generate_stochastic_samples.py at destination: {script_path}\n"
            "Please ensure the file path is correct."
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

    logger.info("Invoking generate_stochastic_samples.py...")
    logger.info(f"Command execution: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        logger.info("Generation routine complete.")
        if result.stdout:
            logger.info(f"Subprocess output:\n{result.stdout}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Generation routine failed: {e}")
        if e.stderr:
            logger.error(f"Subprocess stderr:\n{e.stderr}")
        raise RuntimeError(f"Failed to generate stochastic samples: {e}")

    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Generation script finished execution, but expected output file is missing: {output_file}")

    return load_stochastic_samples_dict(output_file)


def infer_stochastic_file_path(metadata_file: str) -> str:
    base_dir = os.path.dirname(metadata_file)
    return os.path.join(base_dir, "03_stochastic_samples.jsonl")