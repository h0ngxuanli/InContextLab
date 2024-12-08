from typing import List, Optional
from .base import Example, ModelOutput
from .config import ModelConfig
from .amplify_model import AMPLIFYModel
from .semantic_head_model import SemanticHeadModel
from .latent_concept_model import LatentConceptModel
from .information_flow_model import InformationFlowModel

class XICLModels:
    """Unified interface for all ICL models."""
    
    def __init__(self, model_name: str, config: Optional[ModelConfig] = None):
        """Initialize the framework with specified model."""
        self.config = config or ModelConfig()
        self.model = self._initialize_model(model_name)

    def _initialize_model(self, model_name: str):
        """Initialize the appropriate model based on name."""
        from .base import ModelRegistry
        model_class = ModelRegistry.get_model(model_name)
        return model_class(self.config)

    def process_demonstrations(self, examples: List[Example]) -> ModelOutput:
        """Process examples through the selected model."""
        return self.model.process_demonstrations(examples)

    def visualize_results(self, output: ModelOutput) -> None:
        """Visualize results using model-specific visualization."""
        self.model.visualize_results(output)
