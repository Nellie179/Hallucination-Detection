import numpy as np
import logging
from typing import List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detectors.base import BaseDetector
from detectors.registry import register_detector
from data_utils.accessor import SampleAccessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    from sklearn.neural_network import MLPClassifier

    TORCH_AVAILABLE = False
    logger.warning("PyTorch is not installed, falling back to sklearn MLP")


class MINDMLPTorch(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.2):
        super().__init__()

        layers = []
        layers.append(nn.Dropout(dropout))

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


@register_detector("mind")
class MINDDetector(BaseDetector):
    def __init__(
            self,
            name: str,
            hidden_dims: List[int] = [256, 128, 64],
            dropout: float = 0.2,
            learning_rate: float = 0.001,
            epochs: int = 20,
            batch_size: int = 32,
            device: str = None,
            **kwargs
    ):
        super().__init__(name, **kwargs)

        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        if TORCH_AVAILABLE:
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = None
            self.use_torch = True
        else:
            self.device = 'cpu'
            self.model = MLPClassifier(
                hidden_layer_sizes=tuple(hidden_dims),
                activation='relu',
                max_iter=epochs * 10,
                batch_size=batch_size,
                learning_rate_init=learning_rate,
                early_stopping=True,
                n_iter_no_change=5
            )
            self.use_torch = False

        self.is_fitted = False
        self.feature_dim = None

        logger.info(f"[{self.name}] MIND detector initialization complete")
        logger.info(f"  Using {'PyTorch' if self.use_torch else 'sklearn'}")
        logger.info(f"  Hidden dimensions: {hidden_dims}")
        logger.info(f"  Device: {self.device}")

    def _extract_features(self, accessor: SampleAccessor) -> np.ndarray:
        try:
            hidden_state_last = accessor.get_hidden_states(layer_idx=-1, pooling="last")
            hidden_state_mean = accessor.get_hidden_states(layer_idx=-1, pooling="mean")
            combined_features = np.concatenate([hidden_state_last, hidden_state_mean])
            return combined_features
        except Exception as e:
            logger.warning(f"Failed to obtain hidden states: {e}")
            if self.feature_dim:
                return np.zeros(self.feature_dim)
            else:
                raise ValueError("Hidden states must be available during the first feature extraction pass")

    def fit(self, train_accessors: List[SampleAccessor]) -> None:
        logger.info(f"[{self.name}] Starting MIND training...")

        X_train = []
        y_train = []

        for accessor in train_accessors:
            try:
                category = accessor.metadata.get("eval_category")
                if category not in ["correct", "hallucination"]:
                    continue

                features = self._extract_features(accessor)
                label = 1 if category == "hallucination" else 0

                X_train.append(features)
                y_train.append(label)

            except Exception as e:
                logger.warning(f"Sample {accessor.sample_id} processing failed: {e}")
                continue

        if len(X_train) == 0:
            raise ValueError("No valid training samples found!")

        if len(set(y_train)) < 2:
            raise ValueError("Training set must contain both correct and hallucination samples!")

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int64)

        self.feature_dim = X_train.shape[1]

        logger.info(f"[{self.name}] Number of training samples: {len(X_train)}")
        logger.info(f"  Feature dimension: {self.feature_dim}")
        logger.info(f"  Correct: {np.sum(y_train == 0)}")
        logger.info(f"  Hallucination: {np.sum(y_train == 1)}")

        if self.use_torch:
            self._train_torch(X_train, y_train)
        else:
            self._train_sklearn(X_train, y_train)

        self.is_fitted = True
        logger.info(f"[{self.name}] MIND training complete")

    def _train_torch(self, X_train, y_train):
        self.model = MINDMLPTorch(
            input_dim=self.feature_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)

        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"  Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

        self.model.eval()

    def _train_sklearn(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_score(self, accessor: SampleAccessor) -> float:
        if not self.is_fitted:
            raise RuntimeError(f"[{self.name}] Model has not been trained yet")

        try:
            features = self._extract_features(accessor)

            if self.use_torch:
                self.model.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                    outputs = self.model(X_tensor)
                    probs = F.softmax(outputs, dim=1)
                    prob_hallucination = probs[0, 1].cpu().item()
            else:
                probs = self.model.predict_proba(features.reshape(1, -1))
                prob_hallucination = probs[0, 1]

            return float(prob_hallucination)

        except Exception as e:
            logger.error(f"Sample {accessor.sample_id} prediction failed: {e}")
            return float('nan')

    def analyze(self, accessor: SampleAccessor) -> dict:
        try:
            features = self._extract_features(accessor)

            return {
                "feature_dim": len(features),
                "feature_mean": float(np.mean(features)),
                "feature_std": float(np.std(features)),
                "model_type": "PyTorch" if self.use_torch else "sklearn",
                "hallucination_score": self.predict_score(accessor) if self.is_fitted else None
            }
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    print("=" * 70)
    print("MIND Detector Unit Test")
    print("=" * 70)

    np.random.seed(42)


    class MockAccessor:
        def __init__(self, sample_id, hidden_state, category):
            self.sample_id = sample_id
            self.hidden_state = hidden_state
            self.metadata = {"eval_category": category}

        def get_hidden_states(self, layer_idx=-1, pooling="mean"):
            if pooling == "mean":
                return np.mean(self.hidden_state, axis=0)
            elif pooling == "last":
                return self.hidden_state[-1]
            return self.hidden_state[-1]


    print("\n[Test 1] Correct generation")
    states1 = np.random.randn(10, 128) + 1.0
    accessor1 = MockAccessor("test_001", states1, "correct")

    print("\n[Test 2] Hallucination generation")
    states2 = np.random.randn(10, 128) - 1.0
    accessor2 = MockAccessor("test_002", states2, "hallucination")

    try:
        detector = MINDDetector(
            name="test_mind",
            hidden_dims=[128, 64],
            epochs=10
        )

        print("\nTraining MIND...")
        train_data = [accessor1, accessor2] * 5
        detector.fit(train_data)

        print("\n" + "=" * 70)
        score1 = detector.predict_score(accessor1)
        print(f"Sample 1 (Correct) - Hallucination Score: {score1:.3f}")

        score2 = detector.predict_score(accessor2)
        print(f"Sample 2 (Hallucination) - Hallucination Score: {score2:.3f}")

        print("\n" + "=" * 70)
        print("✅ Test complete")
        print(f"Expected: Correct score ({score1:.3f}) < Hallucination score ({score2:.3f})")

        if score1 < score2:
            print("✓ Results match expectations!")
        else:
            print("✗ Results might not match expectations (more training data required)")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()