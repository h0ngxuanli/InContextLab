from typing import List, Optional
from .base import Example, ModelOutput
from .config import ModelConfig
from .amplify_model import AMPLIFYModel
from .semantic_head_model import SemanticHeadModel
from .latent_concept_model import LatentConceptModel
from .information_flow_model import InformationFlowModel

class ICLFramework:
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

    def process_examples(self, examples: List[Example]) -> ModelOutput:
        """Process examples through the selected model."""
        return self.model.process_examples(examples)

    def visualize_results(self, output: ModelOutput) -> None:
        """Visualize results using model-specific visualization."""
        self.model.visualize_results(output)

# Example usage:
def main():
    # Configuration
    config = ModelConfig(
        model_name="gpt2",
        max_length=512,
        batch_size=16
    )

    # Example data in unified format
    examples = [
        Example(
            text="The pen is used for writing.",
            labels=[("pen", "Used-for", "writing")],
            metadata=Metadata(
                task_type=TaskType.RELATION_EXTRACTION,
                split="train"
            )
        ),
        Example(
            text="The movie was fantastic.",
            labels=["positive"],
            metadata=Metadata(
                task_type=TaskType.CLASSIFICATION,
                task_name="sentiment",
                split="demonstration"
            )
        )
    ]

    # Initialize framework with desired model
    framework = ICLFramework("amplify", config)
    
    # Process examples
    output = framework.process_examples(examples)
    
    # Visualize results
    framework.visualize_results(output)

if __name__ == "__main__":
    main()