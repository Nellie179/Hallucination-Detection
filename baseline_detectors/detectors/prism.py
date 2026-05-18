import numpy as np
import logging
from typing import List
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detectors.base import BaseDetector
from detectors.registry import register_detector
from data_utils.accessor import SampleAccessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PRISM_MLP(nn.Module):
    def __init__(self, input_size, dropout=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)


class PRISMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.halu_num = int(self.y.sum().item())

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"input": self.X[idx], "y": self.y[idx]}


@register_detector("prism")
class PRISMDetector(BaseDetector):
    def __init__(self, name: str, target_layer: int = -1, use_prompt_ensemble: bool = False, **kwargs):
        super().__init__(name, **kwargs)
        self.requires_qa_features = True
        self.target_layer = target_layer
        self.use_prompt_ensemble = use_prompt_ensemble

        self.epochs = 10
        self.batch_size = 32
        self.lr = 1e-3
        self.wd = 0.0
        self.dropout = 0.2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.probe = None
        self.is_fitted = False

    def _extract_hidden_state(self, accessor: SampleAccessor) -> np.ndarray:
        if getattr(accessor, "qa_h5_file", None) is None:
            raise ValueError(f"Sample {accessor.sample_id} is missing QA features!")

        grp_name = f"{accessor.sample_id}_prism"
        if grp_name not in accessor.qa_h5_file:
            raise KeyError(f"PRISM group for sample {accessor.sample_id} is missing in H5 file.")

        grp = accessor.qa_h5_file[grp_name]

        layer_str = f"layer_{self.target_layer}"
        if self.target_layer < 0:
            layers = [int(k.split("_")[1]) for k in grp.keys() if k.startswith("layer_")]
            layer_str = f"layer_{max(layers)}"

        feat = np.array(grp[layer_str], dtype=np.float32)
        return feat

    def fit(self, train_accessors: List[SampleAccessor]) -> None:
        logger.info(f"[{self.name}] Starting official 4-layer MLP probe training...")
        X_all, y_all = [], []

        for accessor in train_accessors:
            try:
                category = accessor.metadata.get("eval_category")
                if category not in ["correct", "hallucination"]: continue
                X_all.append(self._extract_hidden_state(accessor))
                y_all.append(1 if category == "hallucination" else 0)
            except Exception:
                continue

        if len(X_all) < 10:
            logger.warning(f"[{self.name}] Too few training samples, skipping training.")
            return

        X_all = np.array(X_all)
        y_all = np.array(y_all)
        input_size = X_all.shape[-1]

        self.probe = PRISM_MLP(input_size=input_size, dropout=self.dropout).to(self.device)

        full_dataset = PRISMDataset(X_all, y_all)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_data, val_data = random_split(full_dataset, [train_size, val_size],
                                            generator=torch.Generator().manual_seed(0))

        train_y = torch.tensor([dataset['y'].item() for dataset in train_data])
        train_halu_num = train_y.sum().item()

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

        nSamples = [train_size - train_halu_num, train_halu_num]
        if sum(nSamples) == 0 or nSamples[0] == 0 or nSamples[1] == 0:
            normedWeights = torch.FloatTensor([1.0, 1.0]).to(self.device)
        else:
            normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
            normedWeights = torch.FloatTensor(normedWeights).to(self.device)

        loss_func = nn.CrossEntropyLoss(weight=normedWeights)

        no_decay = ['bias']
        named_params = list(self.probe.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': self.wd,
             'lr': self.lr},
            {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': self.lr}
        ]
        optimizer = torch.optim.Adam(optimizer_grouped_parameters)

        best_val_acc = -1.0
        best_model_state = None

        for epoch in range(1, self.epochs + 1):
            self.probe.train()
            for batch in train_loader:
                inputs = batch["input"].to(self.device)
                labels = batch["y"].to(self.device)

                optimizer.zero_grad()
                logits = self.probe(inputs)
                loss = loss_func(logits, labels)
                loss.backward()
                optimizer.step()

            self.probe.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["input"].to(self.device)
                    labels = batch["y"]
                    logits = self.probe(inputs)
                    _, preds = torch.max(logits, dim=1)
                    val_preds.extend(preds.cpu().tolist())
                    val_labels.extend(labels.tolist())

            from sklearn.metrics import accuracy_score
            val_acc = accuracy_score(val_labels, val_preds) if val_labels else 0.0

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(self.probe.state_dict())

        if best_model_state is not None:
            self.probe.load_state_dict(best_model_state)

        self.probe.eval()
        self.is_fitted = True
        logger.info(f"[{self.name}] Training complete. Best validation accuracy: {best_val_acc * 100:.2f}%")

    def predict_score(self, accessor: SampleAccessor) -> float:
        if not self.is_fitted: return float('nan')
        try:
            feature = self._extract_hidden_state(accessor)
            feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.probe(feature_tensor)
                prob = torch.softmax(logits, dim=1)[0, 1].item()

            return float(prob)
        except Exception as e:
            return float('nan')