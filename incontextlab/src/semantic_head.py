import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
from .base import BaseICLModel, ModelRegistry, Example, ModelOutput
from .config import ModelConfig, setup_logging

logger = setup_logging(__name__)

def clean_tokens(tokens: List[str]) -> List[str]:
    """Clean special tokens from tokenizer output."""
    return [token.lstrip('Ġ') for token in tokens if len(token.lstrip('Ġ')) != 0]

@ModelRegistry.register("semantic_head")
class SemanticHeadModel(BaseICLModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config.model_name, config.device)
        self.config = config
        self.setup_model()
        self.semantic_factors = {
            'Part-of': 1.2,
            'Compare': 1.0,
            'Used-for': 1.1,
            'Feature-of': 1.0,
            'Hyponym-of': 1.3,
            'Evaluate-for': 1.0,
            'Conjunction': 0.9
        }

    def setup_model(self):
        """Initialize model components."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            output_attentions=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device)
        self.model.eval()

    def get_layer_weight(self, layer_idx: int, weight_name: str) -> torch.Tensor:
        """Get layer weights in a flexible way."""
        return self.model.get_submodule(f'transformer.h.{layer_idx}.attn').get_parameter(weight_name)

    def process_examples(self, examples: List[Example]) -> ModelOutput:
        """Process examples through the Semantic Head pipeline."""
        relation_indices_list = []
        attention_patterns = []
        
        for example in examples:
            # Tokenize input
            tokens = self.tokenizer(
                example.text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.config.max_length
            ).to(self.device)

            # Get attention patterns
            attentions, ov_circuits = self.analyze_attention_heads(
                tokens["input_ids"],
                tokens["attention_mask"]
            )

            # Convert labels to triplets format if needed
            triplets = example.labels if isinstance(example.labels[0], tuple) else []

            # Compute relation indices
            relation_indices = self.compute_relation_index(
                attentions,
                ov_circuits,
                triplets,
                tokens["input_ids"],
                relation_type="semantic"
            )
            relation_indices_list.append(relation_indices)
            attention_patterns.append(attentions)

        # Average results
        mean_relation_indices = np.mean(relation_indices_list, axis=0)

        return ModelOutput(
            scores={},  # No explicit scores for this model
            explanations={
                "relation_indices": mean_relation_indices,
                "attention_patterns": attention_patterns
            },
            visualizations={"heatmap_data": mean_relation_indices},
            additional_info={"token_contributions": self.get_token_contributions(examples)}
        )

    def analyze_attention_heads(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Analyze attention patterns from model."""
        outputs = self.model(
            input_ids, 
            attention_mask=attention_mask, 
            output_attentions=True
        )
        attentions = outputs.attentions
        
        # Get OV circuits for all layers
        ov_circuits = []
        for i in range(len(attentions)):
            ov_circuits.append(self.get_layer_weight(i, 'c_proj.weight'))
            
        return attentions, ov_circuits

    def compute_relation_index(
        self, 
        attentions: List[torch.Tensor], 
        ov_circuits: List[torch.Tensor], 
        triplets: List[Tuple[str, str, str]], 
        input_ids: torch.Tensor, 
        relation_type: str,
        tau: float = 1e-4
    ) -> np.ndarray:
        """Compute relation indices for attention heads."""
        vocab_matrix = self.model.get_output_embeddings().weight
        num_layers = len(attentions)
        num_heads = attentions[0].size(1)
        relation_indices = np.zeros((num_layers, num_heads))
        
        tokens = [self.tokenizer.decode(t) for t in input_ids[0].cpu().numpy()]
        
        for layer_idx, layer_attn in enumerate(attentions):
            for head_idx in range(num_heads):
                head_scores = []
                
                for triplet in triplets:
                    head, rel, tail = triplet
                    
                    # Find token positions
                    head_positions = [i for i, t in enumerate(tokens) if head in t]
                    tail_positions = [i for i, t in enumerate(tokens) if tail in t]
                    
                    if not head_positions or not tail_positions:
                        continue
                    
                    head_idx_pos = sum(head_positions) / len(head_positions)
                    tail_idx_pos = sum(tail_positions) / len(tail_positions)

                    # Get attention weights
                    attention_weights = layer_attn[0, head_idx, :, :].detach().cpu().numpy()
                    
                    # Apply threshold
                    if attention_weights[int(tail_idx_pos), int(head_idx_pos)] / np.max(attention_weights[int(tail_idx_pos), :]) <= tau:
                        continue
                    
                    # Compute OV influence
                    ov_influence = torch.matmul(ov_circuits[layer_idx][head_idx], vocab_matrix.T)
                    ov_influence = ov_influence / ov_influence.max()
                    
                    # Calculate score based on relation type
                    qk_score = attention_weights[int(tail_idx_pos), int(head_idx_pos)]
                    ov_score = ov_influence[int(tail_idx_pos)].item()
                    
                    if relation_type == "semantic":
                        score = qk_score * ov_score * self.get_semantic_factor(rel)
                    else:
                        score = qk_score * ov_score
                    
                    head_scores.append(score)

                # Compute final score for head
                if head_scores:
                    relation_indices[layer_idx, head_idx] = np.mean(head_scores)

        # Normalize scores
        if relation_indices.max() > relation_indices.min():
            relation_indices = (relation_indices - relation_indices.min()) / (relation_indices.max() - relation_indices.min())
        
        return relation_indices

    def get_semantic_factor(self, relation: str) -> float:
        """Return a factor based on the semantic relation type."""
        return self.semantic_factors.get(relation, 1.0)

    def get_token_contributions(
        self, 
        examples: List[Example]
    ) -> Dict[str, List[Dict[str, float]]]:
        """Compute token contributions for each example."""
        contributions = {}
        
        for example in examples:
            sequence = example.text
            inputs = self.tokenizer(
                sequence, 
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length
            ).to(self.device)
            
            embed_layer = self.model.get_input_embeddings()
            input_embeddings = embed_layer(inputs["input_ids"]).clone().detach().requires_grad_(True)

            # Forward pass with embeddings
            outputs = self.model(
                inputs_embeds=input_embeddings,
                output_attentions=True,
                output_hidden_states=True
            )
            
            attentions = outputs.attentions
            logits = outputs.logits

            # Compute gradients
            logits_sum = logits.sum()
            token_grads = torch.autograd.grad(
                outputs=logits_sum,
                inputs=input_embeddings,
                retain_graph=True
            )[0]

            # Combine attention and gradient information
            layer_contributions = []
            for layer_attention in attentions:
                avg_attention = layer_attention.mean(dim=1)
                layer_contributions.append(avg_attention * token_grads.norm(dim=-1))

            combined_contributions = torch.stack(layer_contributions).mean(dim=0)
            contribution_scores = combined_contributions.sum(dim=-1).detach().cpu().numpy()
            contribution_scores = contribution_scores / contribution_scores.sum()

            tokens = clean_tokens(
                self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))
            )

            contributions[example.text] = [
                {"token": token, "score": float(score)}
                for token, score in zip(tokens, contribution_scores[0])
            ]

        return contributions

    def visualize_results(self, output: ModelOutput) -> None:
        """Visualize the results using the heatmap visualization."""
        from .config import Visualizer
        
        # Create heatmap of relation indices
        heatmap_data = output.visualizations["heatmap_data"]
        Visualizer.create_heatmap(
            data=heatmap_data,
            title="Semantic Head Analysis - Relation Indices"
        )
        
        # Create token contribution visualization if available
        if "token_contributions" in output.additional_info:
            Visualizer.create_token_visualization(
                data=output.additional_info["token_contributions"],
                title="Token Contributions Analysis"
            )