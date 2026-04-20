# baseline_detectors/data_utils/data_split.py
import os
import json
import argparse
from sklearn.model_selection import train_test_split

def split_dataset(input_jsonl, exp_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    print(f"[*] 正在读取数据集: {input_jsonl}")
    
    valid_samples = []
    skipped = 0
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            cat = data.get("eval_category")
            # 我们只切分有明确标签的数据用于评测
            if cat in ["correct", "hallucination"]:
                valid_samples.append(data)
            else:
                skipped += 1
                
    labels = [1 if d.get("eval_category") == "hallucination" else 0 for d in valid_samples]
    
    print(f"[*] 有效样本数: {len(valid_samples)} (跳过未知标签: {skipped})")
    print(f"[*] 幻觉比例: {sum(labels)/len(labels)*100:.2f}%")

    # 1. 先切出 Train 和 Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        valid_samples, labels, test_size=(val_ratio + test_ratio), stratify=labels, random_state=42
    )
    
    # 2. 再把 Temp 切成 Val 和 Test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, _, _ = train_test_split(
        X_temp, y_temp, test_size=relative_test_ratio, stratify=y_temp, random_state=42
    )
    
    splits = {"train": X_train, "val": X_val, "test": X_test}
    
    for split_name, samples in splits.items():
        out_path = os.path.join(exp_dir, f"03_{split_name}.jsonl")
        with open(out_path, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        
        pos = sum(1 for s in samples if s.get("eval_category") == "hallucination")
        print(f"[+] {split_name.upper():5} 写入成功 | 数量: {len(samples)} | 幻觉率: {pos/len(samples)*100:.2f}% -> {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--exp_dir", type=str, required=True)
    args = parser.parse_args()
    split_dataset(args.input_jsonl, args.exp_dir)