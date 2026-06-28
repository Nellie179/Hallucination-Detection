import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from typing import List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detectors.base import BaseDetector
from detectors.registry import register_detector
from data_utils.accessor import SampleAccessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ICRProbe(nn.Module):
    """
    MLP probe matching the original ICR paper architecture.
    input_dim -> 128 -> 64 -> 32 -> 1
    with BatchNorm1d, LeakyReLU(0.01), Dropout(0.3), Kaiming init.
    """
    def __init__(self, input_dim: int):
        super(ICRProbe, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.01, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        out = self.dropout1(out)
        out = F.leaky_relu(self.bn2(self.fc2(out)), negative_slope=0.01)
        out = self.dropout2(out)
        out = F.leaky_relu(self.bn3(self.fc3(out)), negative_slope=0.01)
        out = self.dropout3(out)
        return self.sigmoid(self.fc4(out))


@register_detector("icr_probe")
class ICRProbeDetector(BaseDetector):
    def __init__(
            self,
            name: str = "icr_probe",
            use_icr_only: bool = True,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            epochs: int = 50,
            batch_size: int = 32,
            val_split: float = 0.2,
            lr_factor: float = 0.5,
            lr_patience: int = 5,
            **kwargs
    ):
        super().__init__(name, **kwargs)

        self.requires_qa_features = True

        self.use_icr_only = use_icr_only
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_split = val_split
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.is_fitted = False

        logger.info(f"[{self.name}] ICR Probe initialization complete")

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
        logger.info(f"[{self.name}] Starting ICR Probe training...")

        X_list, y_list = [], []
        for acc in train_accessors:
            try:
                category = acc.metadata.get("eval_category")
                if category not in ["correct", "hallucination"]:
                    continue
                feat = self._get_feature(acc)
                X_list.append(feat)
                y_list.append(1 if category == "hallucination" else 0)
            except Exception:
                continue

        if len(set(y_list)) < 2:
            logger.warning(f"[{self.name}] Insufficient labels in training set, skipping.")
            return

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        # Train / val split
        n_val = max(1, int(len(X) * self.val_split))
        n_train = len(X) - n_val
        indices = np.random.permutation(len(X))
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        X_train = torch.tensor(X[train_idx]).to(self.device)
        y_train = torch.tensor(y[train_idx]).to(self.device)
        X_val   = torch.tensor(X[val_idx]).to(self.device)
        y_val   = torch.tensor(y[val_idx]).to(self.device)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=self.batch_size,
            shuffle=False
        )

        self.model = ICRProbe(input_dim=X.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=self.lr_factor, patience=self.lr_patience
        )
        criterion = nn.BCELoss()

        best_val_loss = float('inf')
        best_state = None

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = criterion(self.model(X_batch), y_batch.unsqueeze(1))
                loss.backward()
                optimizer.step()

            # Validate
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    loss = criterion(self.model(X_batch), y_batch.unsqueeze(1))
                    val_losses.append(loss.item())
            val_loss = np.mean(val_losses)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                logger.info(f"[{self.name}] Epoch {epoch+1}/{self.epochs} | val_loss: {val_loss:.4f}")

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        self.model.eval()
        self.is_fitted = True
        logger.info(f"[{self.name}] Training complete. Best val_loss: {best_val_loss:.4f}")

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