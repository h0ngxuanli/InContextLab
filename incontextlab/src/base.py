from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum

class TaskType(Enum):
    CLASSIFICATION = "classification"
    RELATION_EXTRACTION = "relation_extraction"

@dataclass
class Metadata:
    task_type: TaskType
    task_name: Optional[str] = None
    demonstration_type: Optional[str] = None
    split: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None

@dataclass
class Example:
    text: str
    labels: Union[List[str], List[Tuple[str, str, str]]]
    metadata: Optional[Metadata] = None

@dataclass
class ModelOutput:
    scores: Dict[str, float]
    explanations: Optional[Dict[str, Any]] = None
    visualizations: Optional[Dict[str, Any]] = None
    additional_info: Optional[Dict[str, Any]] = None

class BaseICLModel:
    """Base class for all ICL models."""
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device

    def process_demonstrations(self, examples: List[Example]) -> ModelOutput:
        raise NotImplementedError

    def visualize_results(self, output: ModelOutput) -> None:
        raise NotImplementedError

class ModelRegistry:
    """Registry for available models."""
    _models = {}

    @classmethod
    def register(cls, name: str):
        def decorator(model_class):
            cls._models[name] = model_class
            return model_class
        return decorator

    @classmethod
    def get_model(cls, name: str):
        if name not in cls._models:
            raise ValueError(f"Model {name} not found. Available models: {list(cls._models.keys())}")
        return cls._models[name]