import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
from torch.optim import AdamW
from .base import BaseICLModel, ModelRegistry, Example, ModelOutput
from .config import ModelConfig, setup_logging


logger = setup_logging(__name__)

class CausalDirection(Enum):
    """Enumeration for causal direction in demonstration construction."""
    X_TO_Y = "x_to_y"  # Input to label direction
    Y_TO_X = "y_to_x"  # Label to input direction

def set_global_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

def clean_tokens(tokens: List[str]) -> List[str]:
    """Clean special tokens from tokenizer output."""
    return [token.lstrip('Ġ') for token in tokens if len(token.lstrip('Ġ')) != 0]

@ModelRegistry.register("latent_concept")
class LatentConceptModel(BaseICLModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config.model_name, config.device)
        self.config = config
        set_global_seed(42)
        self.setup_model()

    def setup_model(self):
        """Initialize model components."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            output_hidden_states=True,
            output_attentions=True
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.resize_token_embeddings(len(self.tokenizer))

    def add_concept_tokens(self, tasks: List[str]) -> List[str]:
        """Add concept tokens for each task."""
        new_tokens = [f"<{task}_concept>" for task in tasks]
        self.tokenizer.add_tokens(new_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Initialize new token embeddings
        embed_layer = self.model.get_input_embeddings()
        new_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in new_tokens]
        
        with torch.no_grad():
            mean_embedding = embed_layer.weight.mean(dim=0)
            for token_id in new_token_ids:
                embed_layer.weight[token_id] = mean_embedding

        return new_tokens

    def train_concept_tokens(
        self, 
        train_data: List[Dict], 
        tasks: List[str], 
        direction: CausalDirection
    ):
        """Train concept token embeddings."""
        # Add semantic concept tokens
        concept_tokens = self.add_concept_tokens(tasks)
        embed_layer = self.model.get_input_embeddings()

        # Optimizer for concept token embeddings
        optimizer = AdamW([{
            'params': embed_layer.weight,
            'lr': self.config.concept_learning_rate
        }])
        
        new_token_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(token) for token in concept_tokens],
            device=self.device
        )

        for step in range(self.config.concept_train_steps):
            # Create batch
            batch = random.sample(
                train_data,
                min(self.config.concept_batch_size, len(train_data))
            )
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)

            # Compute batch loss
            for item in batch:
                loss = self._compute_batch_loss(item, concept_tokens, direction)
                total_loss += loss / len(batch)
            
            # Backpropagate
            total_loss.backward()

            # Regularize gradients for concept tokens
            with torch.no_grad():
                concept_grads = embed_layer.weight.grad[new_token_ids]
                concept_grad_norms = concept_grads.norm(dim=-1, keepdim=True)
                embed_layer.weight.grad[new_token_ids] /= concept_grad_norms.clamp(min=1e-8)

            optimizer.step()

            if step % 200 == 0:
                logger.info(f"Step {step}, Loss: {total_loss.item()}")

    def _compute_batch_loss(
        self, 
        item: Dict, 
        concept_tokens: List[str], 
        direction: CausalDirection
    ) -> torch.Tensor:
        """Compute loss for a single batch item."""
        concept_token = concept_tokens[0]  # Use first concept token

        # Construct sequence based on direction
        if direction == CausalDirection.X_TO_Y:
            sequence = f"{concept_token} {item['text']} {item['label']} {self.tokenizer.eos_token}"
        else:
            sequence = f"{concept_token} {item['label']} {item['text']} {self.tokenizer.eos_token}"

        # Tokenize and compute loss
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)
        
        target_ids = inputs["input_ids"]
        outputs = self.model(**inputs)
        
        # Compute language modeling loss
        shift_logits = outputs.logits[..., :-1, :]
        shift_labels = target_ids[..., 1:]
        loss_fct = torch.nn.CrossEntropyLoss()
        
        return loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

    def score_demonstrations(
        self, 
        pool: List[Dict], 
        concept_token: str, 
        test_input: str
    ) -> List[Dict]:
        """Score demonstrations based on their relevance to test input."""
        scores = []
        for demo in pool:
            sequence = f"{concept_token} {demo['text']} {concept_token} {test_input} {self.tokenizer.eos_token}"
            inputs = self.tokenizer(
                sequence,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length
            ).to(self.device)
            
            embed_layer = self.model.get_input_embeddings()
            input_embeddings = embed_layer(inputs["input_ids"]).clone().detach().requires_grad_(True)

            outputs = self.model(inputs_embeds=input_embeddings, output_attentions=True)
            logits = outputs.logits

            logits_sum = logits.sum()
            token_grads = torch.autograd.grad(
                outputs=logits_sum,
                inputs=input_embeddings,
                retain_graph=True
            )[0]
            
            demo_score = token_grads.norm(dim=-1).sum().item()
            scores.append({"demo": demo, "score": demo_score})

        total_score = sum(score["score"] for score in scores)
        for score in scores:
            score["score"] /= total_score

        return sorted(scores, key=lambda x: x["score"], reverse=True)

    def compute_token_contributions(
        self, 
        sequence: str,
        test_input: Optional[str] = None
    ) -> List[Dict]:
        """Compute token contributions through gradient analysis."""
        if test_input:
            sequence = f"{sequence} {self.tokenizer.eos_token} {test_input}"
            
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)
        
        embed_layer = self.model.get_input_embeddings()
        input_embeddings = embed_layer(inputs["input_ids"]).clone().detach().requires_grad_(True)

        outputs = self.model(
            inputs_embeds=input_embeddings,
            output_attentions=True,
            output_hidden_states=True
        )
        
        attentions = outputs.attentions
        logits = outputs.logits

        logits_sum = logits.sum()
        token_grads = torch.autograd.grad(
            outputs=logits_sum,
            inputs=input_embeddings,
            retain_graph=True
        )[0]

        layer_contributions = []
        for layer_attention in attentions:
            avg_attention = layer_attention.mean(dim=1)
            layer_contributions.append(avg_attention * token_grads.norm(dim=-1))

        combined_contributions = torch.stack(layer_contributions).mean(dim=0)
        contributions = combined_contributions.sum(dim=-1).detach().cpu().numpy()
        contributions /= contributions.sum()

        tokens = clean_tokens(
            self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))
        )

        return [
            {"token": token, "score": float(contribution)}
            for token, contribution in zip(tokens, contributions[0])
        ]

    def select_top_k_demonstrations(
        self,
        pool: List[Dict],
        concept_token: str,
        test_input: str,
        k: int
    ) -> List[Dict]:
        """Select top-k demonstrations based on scores."""
        scored_demos = self.score_demonstrations(pool, concept_token, test_input)
        selected_demos = scored_demos[:k]
        
        for demo in selected_demos:
            sequence = f"{concept_token} {demo['demo']['text']}"
            demo['token_contributions'] = self.compute_token_contributions(
                sequence,
                test_input=test_input
            )
        
        return selected_demos

    def process_demonstrations(self, examples: List[Example]) -> ModelOutput:
        """Process examples through the Latent Concept pipeline."""
        tasks = list(set(ex.metadata.task_name for ex in examples if ex.metadata))
        
        train_data = [
            {"text": ex.text, "label": ex.labels[0], "task": ex.metadata.task_name}
            for ex in examples if ex.metadata and ex.metadata.split == "train"
        ]
        
        concept_tokens = self.add_concept_tokens(tasks)
        self.train_concept_tokens(
            train_data,
            tasks=tasks,
            direction=CausalDirection.X_TO_Y
        )

        demo_pool = [ex for ex in examples if ex.metadata and ex.metadata.split == "demonstration_pool"]
        scored_demos = []
        token_contributions = {}
        
        for test_ex in examples:
            if test_ex.metadata and test_ex.metadata.split == "test":
                demos = self.select_top_k_demonstrations(
                    [{"text": d.text, "label": d.labels[0]} for d in demo_pool],
                    concept_tokens[0],
                    test_ex.text,
                    k=self.config.top_k_demons
                )
                
                for demo in demos:
                    sequence = f"{concept_tokens[0]} {demo['demo']['text']}"
                    contributions = self.compute_token_contributions(
                        sequence,
                        test_input=test_ex.text
                    )
                    token_contributions[demo['demo']['text']] = contributions
                
                scored_demos.extend(demos)

        return ModelOutput(
            scores={str(i): demo["score"] for i, demo in enumerate(scored_demos)},
            explanations={"token_contributions": token_contributions},
            visualizations={"demonstrations": scored_demos},
            additional_info={"concept_tokens": concept_tokens}
        )

    def visualize_results(self, output: ModelOutput) -> None:
        """Visualize the results using the dashboard visualizer."""
        from .config import Visualizer
        Visualizer.create_latent_concept_visualization(
            output
        )