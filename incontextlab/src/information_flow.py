import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
from .base import BaseICLModel, ModelRegistry, Example, ModelOutput
from .config import ModelConfig, setup_logging

logger = setup_logging(__name__)

class SaliencyAnalyzer:
    """Analyzes saliency patterns in transformer models."""
    
    def compute_saliency_scores(
        self, 
        attention_matrices: List[torch.Tensor], 
        loss: torch.Tensor
    ) -> List[torch.Tensor]:
        """Compute saliency scores for attention matrices."""
        attention_matrices = [
            A.clone().detach().requires_grad_(True) 
            for A in attention_matrices
        ]
        
        # Combine attention matrices for gradient computation
        combined_matrix = sum([A.sum() for A in attention_matrices])
        augmented_loss = loss + combined_matrix
        augmented_loss.backward(retain_graph=True)
        
        saliency_scores = []
        for A in attention_matrices:
            grad = A.grad
            if grad is None:
                raise ValueError("Gradient not computed for attention matrix")
            saliency = torch.abs(A * grad)
            saliency_scores.append(saliency)
        
        return saliency_scores
    
    def compute_flow_metrics(
        self,
        saliency_scores: List[torch.Tensor],
        class_positions: torch.Tensor,
        target_position: int,
        normalize: bool = True
    ) -> Dict[str, List[float]]:
        """Compute flow metrics from saliency scores."""
        raw_metrics = {"S_wp": [], "S_pq": [], "S_ww": []}
        
        for saliency in saliency_scores:
            # Compute text-to-label connections
            wp_connections = self._get_text_to_label_connections(class_positions)
            S_wp = self._compute_average_flow(saliency, wp_connections)
            
            # Compute label-to-target connections
            pq_connections = self._get_label_to_target_connections(class_positions, target_position)
            S_pq = self._compute_average_flow(saliency, pq_connections)
            
            # Compute other connections
            ww_connections = self._get_other_connections(class_positions, target_position, saliency.shape[-1])
            S_ww = self._compute_average_flow(saliency, ww_connections)
            
            raw_metrics["S_wp"].append(S_wp)
            raw_metrics["S_pq"].append(S_pq)
            raw_metrics["S_ww"].append(S_ww)
        
        if normalize:
            normalized_metrics = {}
            for key, values in raw_metrics.items():
                min_val, max_val = min(values), max(values)
                if max_val > min_val:
                    normalized_metrics[key] = [
                        (v - min_val) / (max_val - min_val) 
                        for v in values
                    ]
                else:
                    normalized_metrics[key] = [0.0 for _ in values]
            return normalized_metrics
        
        return raw_metrics

    def _get_text_to_label_connections(
        self, 
        class_positions: torch.Tensor
    ) -> List[Tuple[int, int]]:
        """Get connections from text tokens to label tokens."""
        return [
            (p_k.item(), j) 
            for p_k in class_positions 
            for j in range(p_k.item())
        ]

    def _get_label_to_target_connections(
        self, 
        class_positions: torch.Tensor, 
        target_position: int
    ) -> List[Tuple[int, int]]:
        """Get connections from label tokens to target token."""
        return [
            (target_position, p_k.item()) 
            for p_k in class_positions
        ]

    def _get_other_connections(
        self,
        class_positions: torch.Tensor,
        target_position: int,
        seq_len: int
    ) -> List[Tuple[int, int]]:
        """Get all other token-to-token connections."""
        all_pairs = [
            (i, j) 
            for i in range(seq_len) 
            for j in range(i)
        ]
        exclude = set(
            self._get_text_to_label_connections(class_positions) +
            self._get_label_to_target_connections(class_positions, target_position)
        )
        return list(set(all_pairs) - exclude)

    def _compute_average_flow(
        self,
        saliency: torch.Tensor,
        connections: List[Tuple[int, int]]
    ) -> float:
        """Compute average flow strength for given connections."""
        if not connections:
            return 0.0
            
        total = 0.0
        for i, j in connections:
            if i >= saliency.shape[2] or j >= saliency.shape[3]:
                continue
            total += saliency[:, :, i, j].sum().item()
            
        return total / len(connections) if connections else 0.0

@ModelRegistry.register("information_flow")
class InformationFlowModel(BaseICLModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config.model_name, config.device)
        self.config = config
        self.setup_model()
        self.saliency_analyzer = SaliencyAnalyzer()

    def setup_model(self):
        """Initialize model components."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            output_attentions=True
        ).to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.eval()

    def get_class_positions(
        self,
        input_text: str,
        labels: List[str]
    ) -> List[int]:
        """Find positions of class labels in input text."""
        tokenized_input = self.tokenizer(
            input_text,
            return_tensors="pt",
            add_special_tokens=False
        )
        input_ids = tokenized_input["input_ids"][0]
        positions = []
        
        for label in labels:
            # Try exact match
            label_ids = self.tokenizer.encode(label, add_special_tokens=False)
            found_position = None
            
            for idx in range(len(input_ids) - len(label_ids) + 1):
                if torch.all(input_ids[idx:idx + len(label_ids)] == torch.tensor(label_ids)):
                    found_position = idx
                    break
            
            # Try with space prefix
            if found_position is None:
                spaced_label = " " + label if not label.startswith(" ") else label
                label_ids = self.tokenizer.encode(spaced_label, add_special_tokens=False)
                
                for idx in range(len(input_ids) - len(label_ids) + 1):
                    if torch.all(input_ids[idx:idx + len(label_ids)] == torch.tensor(label_ids)):
                        found_position = idx
                        break
            
            if found_position is not None:
                positions.append(found_position)
            else:
                logger.warning(f"Label '{label}' not found in input text.")
        
        return positions

    def find_last_demonstration_position(
        self,
        input_ids: torch.Tensor
    ) -> int:
        """Find the position of the last demonstration token."""
        demo_token_ids = self.tokenizer.encode(
            "Demonstration:",
            add_special_tokens=False
        )
        demo_length = len(demo_token_ids)
        positions = []
        
        for i in range(len(input_ids) - demo_length + 1):
            if torch.all(input_ids[i:i + demo_length] == torch.tensor(demo_token_ids)):
                positions.append(i)
        
        if not positions:
            raise ValueError("No 'Demonstration:' token found in input_ids.")
        
        last_demo_start = positions[-1]
        target_position = last_demo_start + demo_length
        return min(target_position, len(input_ids) - 1)

    def process_examples(self, examples: List[Example]) -> ModelOutput:
        """Process examples through the Information Flow pipeline."""
        demonstration_text = ""
        labels = []
        
        # Build demonstration text and collect labels
        for example in examples:
            if example.metadata and example.metadata.split == "demonstration":
                text = example.text
                label = example.labels[0]
                demonstration_text += f"Demonstration: {text}\nLabel: {label}\n\n"
                labels.append(label)
        
        # Process final example
        final_example = next(
            ex for ex in examples 
            if ex.metadata and ex.metadata.split == "test"
        )
        full_text = f"{demonstration_text}Demonstration: {final_example.text}\nLabel:"

        # Tokenize and prepare inputs
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            return_attention_mask=True
        ).to(self.device)
        
        input_ids = inputs["input_ids"][0]
        target_position = self.find_last_demonstration_position(input_ids)
        
        # Get model outputs
        self.model.train()  # Enable gradient computation
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        
        # Compute saliency scores
        saliency_scores = self.saliency_analyzer.compute_saliency_scores(
            list(outputs.attentions),
            outputs.loss
        )
        
        # Get label positions and compute flow metrics
        class_positions = torch.tensor(
            self.get_class_positions(full_text, labels)
        )
        
        metrics = self.saliency_analyzer.compute_flow_metrics(
            saliency_scores,
            class_positions,
            target_position,
            normalize=True
        )
        
        # Prepare visualization data
        layers = list(range(1, len(metrics["S_wp"]) + 1))
        
        return ModelOutput(
            scores={},  # No explicit scores for this model
            explanations={
                "flow_metrics": metrics,
                "saliency_scores": saliency_scores
            },
            visualizations={
                "flow_data": {
                    "layers": layers,
                    "metrics": metrics
                }
            },
            additional_info={
                "target_position": target_position,
                "class_positions": class_positions.tolist(),
                "attention_patterns": [a.detach().cpu().numpy() for a in outputs.attentions]
            }
        )

    def visualize_results(self, output: ModelOutput) -> None:
        """Visualize the information flow results."""
        from .config import Visualizer
        Visualizer.create_flow_visualization(
            output.visualizations["flow_data"]["layers"],
            output.visualizations["flow_data"]["metrics"],
            "Information Flow Analysis Across Layers"
        )