import numpy as np
import logging
import torch
import torch.nn as nn
from typing import List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detectors.base import BaseDetector
from detectors.registry import register_detector
from data_utils.accessor import SampleAccessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ICR_MLP(nn.Module):
    def __init__(self, input_dim):
        super(ICR_MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


@register_detector("icr_probe")
class ICRProbeDetector(BaseDetector):
    def __init__(
            self,
            name: str = "icr_probe",
            use_icr_only: bool = True,
            learning_rate: float = 1e-3,
            epochs: int = 15,
            **kwargs
    ):
        super().__init__(name, **kwargs)

        self.requires_qa_features = True

        self.use_icr_only = use_icr_only
        self.lr = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.is_fitted = False

        logger.info(
            f"[{self.name}] ICR Probe initialization complete (Mode: {'JS-Divergence' if use_icr_only else 'Mean-Pooling HS'})")

    def _get_feature(self, accessor: SampleAccessor) -> np.ndarray:
        if accessor.qa_h5_file is None:
            raise ValueError(f"Sample {accessor.sample_id} is missing H5 handle")

        grp_name = f"{accessor.sample_id}_icr_probe"
        if grp_name not in accessor.qa_h5_file:
            raise KeyError(f"ICR data group for sample {accessor.sample_id} is missing in H5")

        grp = accessor.qa_h5_file[grp_name]

        if self.use_icr_only:
            if "icr_feature" not in grp:
                raise KeyError(f"Sample {accessor.sample_id} has not extracted icr_feature yet!")
            return np.array(grp["icr_feature"], dtype=np.float32)
        else:
            layers = [k for k in grp.keys() if k.startswith("layer_")]
            last_layer = sorted(layers, key=lambda x: int(x.split("_")[1]))[-1]
            return np.array(grp[last_layer], dtype=np.float32)

    def fit(self, train_accessors: List[SampleAccessor]) -> None:
        logger.info(f"[{self.name}] Starting MLP network training...")

        X_train, y_train = [], []
        for acc in train_accessors:
            try:
                category = acc.metadata.get("eval_category")
                if category not in ["correct", "hallucination"]:
                    continue

                feat = self._get_feature(acc)
                label = 1 if category == "hallucination" else 0

                X_train.append(feat)
                y_train.append(label)
            except Exception as e:
                continue

        if len(set(y_train)) < 2:
            logger.warning(f"[{self.name}] Insufficient labels in the training set, skipping training.")
            return

        X = torch.tensor(np.array(X_train), dtype=torch.float32).to(self.device)
        y = torch.tensor(np.array(y_train), dtype=torch.float32).unsqueeze(1).to(self.device)

        self.model = ICR_MLP(input_dim=X.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        self.is_fitted = True
        logger.info(f"[{self.name}] Training complete (Epochs: {self.epochs})")

    def predict_score(self, accessor: SampleAccessor) -> float:
        if not self.is_fitted or self.model is None:
            return float('nan')

        try:
            feat = self._get_feature(accessor)
            x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)

            self.model.eval()
            with torch.no_grad():
                prob = self.model(x).cpu().item()
            return float(prob)

        except Exception as e:
            logger.error(f"Sample {accessor.sample_id} prediction failed: {e}")
            return float('nan')