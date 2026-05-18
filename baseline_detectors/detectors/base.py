from typing import List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_utils.accessor import SampleAccessor


class BaseDetector:
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs

        self.requires_stochastic: bool = False
        self.requires_qa_features: bool = False
        self.required_qa_method: Optional[str] = None

    def fit(self, train_accessors: List[SampleAccessor]) -> None:
        pass

    def predict_score(self, accessor: SampleAccessor) -> float:
        raise NotImplementedError