import json


def load_labels(label_jsonl_path: str, label_mapping: dict):
    """
    Returns:
        label_dict: {sample_id: int_label}
    """
    label_dict = {}

    with open(label_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            sample_id = item["sample_id"]
            category = item["eval_category"]

            if category not in label_mapping:
                raise ValueError(f"Unknown eval_category '{category}' for sample {sample_id}")

            label_dict[sample_id] = label_mapping[category]

    return label_dict