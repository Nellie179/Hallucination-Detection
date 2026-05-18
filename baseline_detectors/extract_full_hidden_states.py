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

    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        layers: List[int] = None,
        dtype: torch.dtype = torch.float16
    ):
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Device: {device}")

        self.device = device
        self.dtype = dtype
        self.layers = layers

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

        self.num_layers = self.model.config.num_hidden_layers
        logger.info(f"Model layers: {self.num_layers}")

        if self.layers is None:
            self.layers = list(range(self.num_layers))
        logger.info(f"Extraction layers: {self.layers}")

    def extract_hidden_states(
        self,
        question: str,
        answer: str
    ) -> Tuple[List[np.ndarray], int]:
        full_text = question + " " + answer

        question_tokens = self.tokenizer(question, add_special_tokens=True)
        full_tokens = self.tokenizer(full_text, add_special_tokens=True, return_tensors="pt")

        question_length = len(question_tokens['input_ids'])

        input_ids = full_tokens['input_ids'].to(self.device)
        attention_mask = full_tokens['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

        hidden_states_by_layer = {}

        for layer_idx in self.layers:
            hidden = outputs.hidden_states[layer_idx + 1]  
            hidden = hidden.squeeze(0).cpu().numpy()  

            hidden = hidden.astype(np.float16)
            hidden_states_by_layer[layer_idx] = hidden

        return hidden_states_by_layer, question_length

    def extract_batch(
        self,
        samples: List[Dict],
        output_file: str
    ):
        logger.info(f"Starting feature extraction for {len(samples)} samples")
        logger.info(f"Output destination: {output_file}")

        with h5py.File(output_file, 'w') as f:
            for sample in tqdm(samples, desc="Extracting hidden states"):
                try:
                    sample_id = sample['sample_id']
                    question = sample['question']
                    answer = sample['answer']

                    hidden_states_by_layer, question_length = self.extract_hidden_states(
                        question, answer
                    )

                    sample_group = f.create_group(sample_id)

                    sample_group.attrs['question_length'] = question_length
                    sample_group.attrs['answer_start_idx'] = question_length
                    sample_group.attrs['question'] = question
                    sample_group.attrs['answer'] = answer

                    seq_len = list(hidden_states_by_layer.values())[0].shape[0]

                    for token_idx in range(seq_len):
                        token_group = sample_group.create_group(f"token_{token_idx}")

                        for layer_idx, hidden_states in hidden_states_by_layer.items():
                            token_hidden = hidden_states[token_idx, :]  
                            token_group.create_dataset(
                                f"layer_{layer_idx}",
                                data=token_hidden,
                                compression="gzip",
                                compression_opts=4
                            )

                except Exception as e:
                    logger.error(f"Failed to extract features for sample {sample.get('sample_id', 'unknown')}: {e}")
                    continue

        logger.info(f"Extraction execution finished. Saved to: {output_file}")


def load_samples_from_json(input_file: str) -> List[Dict]:
    logger.info(f"Loading target dataset file: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        samples = list(data.values())
    else:
        raise ValueError(f"Unsupported data format configuration: {type(data)}")

    logger.info(f"Loaded {len(samples)} total target samples")
    return samples


def main():
    parser = argparse.ArgumentParser(description='Extract complete Q+A model representations')

    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Input JSON dataset path'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Output HDF5 array path'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Target model name configuration or hub location'
    )
    parser.add_argument(
        '--layers',
        type=str,
        default='all',
        help='Comma-separated targets layer values or "all"'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Execution device profile'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='float16',
        choices=['float16', 'float32', 'bfloat16'],
        help='Tensor precision initialization configuration'
    )

    args = parser.parse_args()

    if args.layers.lower() == 'all':
        layers = None
    else:
        layers = [int(x.strip()) for x in args.layers.split(',')]

    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]

    samples = load_samples_from_json(args.input_file)

    extractor = HiddenStatesExtractor(
        model_name=args.model_name,
        device=args.device,
        layers=layers,
        dtype=dtype
    )

    extractor.extract_batch(samples, args.output_file)

    logger.info("All operations complete.")


if __name__ == "__main__":
    main()