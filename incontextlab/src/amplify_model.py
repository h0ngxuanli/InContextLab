import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk
from dataclasses import dataclass
from .base import BaseICLModel, ModelRegistry, Example, ModelOutput
from .config import ModelConfig, setup_logging

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Warning: Failed to download stopwords: {e}")

logger = setup_logging(__name__)

@dataclass
class Sample:
    """Class for keeping track of a sample with its metadata."""
    text: str
    label: str
    score: Optional[float] = None
    explanation: Optional[List[str]] = None

def clean_token(token: str) -> str:
    """Clean special tokens from GPT-2 tokenizer output."""
    return token.replace('Ġ', '')

@ModelRegistry.register("amplify")
class AMPLIFYModel(BaseICLModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config.model_name, config.device)
        self.config = config
        self.setup_model()

    def setup_model(self):
        """Initialize model components."""
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2ForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2
            ).to(self.device)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
            
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def predict(self, text: str) -> Dict[str, float]:
        """Generate prediction scores."""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            probs = F.softmax(outputs.logits, dim=-1)[0].cpu()
            return {str(i): float(prob) for i, prob in enumerate(probs)}
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return {str(i): 0.0 for i in range(2)}

    def compute_vanilla_gradients(
        self,
        text: str,
        label: str
    ) -> np.ndarray:
        """Generate explanations using vanilla gradients."""
        try:
            self.model.zero_grad()
            
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            embeddings = self.model.transformer.wte(inputs["input_ids"])
            embeddings.retain_grad()
            
            outputs = self.model(inputs_embeds=embeddings)
            target_class = int(label)
            
            outputs.logits[0, target_class].backward()
            
            gradients = embeddings.grad
            if gradients is None:
                return np.zeros(len(inputs["input_ids"][0]))
            
            token_attributions = torch.norm(gradients[0], p=2, dim=-1).cpu().numpy()
            tokens = [clean_token(token) for token in self.tokenizer.tokenize(text)]
            
            # Filter out stopwords
            token_attributions = [
                attr for attr, token in zip(token_attributions, tokens)
                if token and token.lower() not in stopwords.words('english')
            ]
            
            return np.array(token_attributions)
            
        except Exception as e:
            logger.error(f"Error in vanilla gradients: {str(e)}")
            return np.zeros(1)

    def compute_contrastive_gradients(
        self,
        text: str,
        label: str
    ) -> np.ndarray:
        """Generate contrastive explanations."""
        try:
            self.model.zero_grad()
            
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Get predictions first
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_class = outputs.logits.argmax(dim=-1).item()
            
            # Get gradients for true class
            embeddings = self.model.transformer.wte(inputs["input_ids"])
            embeddings.retain_grad()
            
            outputs = self.model(inputs_embeds=embeddings)
            target_class = int(label)
            
            outputs.logits[0, target_class].backward(retain_graph=True)
            true_gradients = embeddings.grad.clone()
            
            # Reset gradients
            self.model.zero_grad()
            embeddings.grad = None
            
            # Second backward pass for predicted class
            outputs = self.model(inputs_embeds=embeddings)
            outputs.logits[0, predicted_class].backward()
            pred_gradients = embeddings.grad.clone()
            
            if true_gradients is None or pred_gradients is None:
                return np.zeros(len(inputs["input_ids"][0]))
            
            # Compute contrastive gradients
            tokens = [clean_token(token) for token in self.tokenizer.tokenize(text)]
            true_grads = [
                attr for attr, token in zip(true_gradients[0], tokens)
                if token and token.lower() not in stopwords.words('english')
            ]
            pred_grads = [
                attr for attr, token in zip(pred_gradients[0], tokens)
                if token and token.lower() not in stopwords.words('english')
            ]
            
            contrastive = np.array(true_grads) - np.array(pred_grads)
            return np.linalg.norm(contrastive, ord=2, axis=-1)
            
        except Exception as e:
            logger.error(f"Error in contrastive gradients: {str(e)}")
            return np.zeros(1)

    def generate_explanation(
        self,
        text: str,
        label: str,
        method: str = "vanilla"
    ) -> np.ndarray:
        """Generate explanations using specified method."""
        if method == "vanilla":
            return self.compute_vanilla_gradients(text, label)
        elif method == "contrastive":
            return self.compute_contrastive_gradients(text, label)
        else:
            logger.warning(f"Unknown method {method}, falling back to vanilla gradients")
            return self.compute_vanilla_gradients(text, label)

    def compute_mcs(self, text: str, true_label: str) -> float:
        """Compute Misclassification Confidence Score."""
        try:
            predictions = self.predict(text)
            true_score = predictions.get(true_label, 0)
            max_false_score = max(
                (score for label, score in predictions.items() if label != true_label),
                default=0
            )
            return max_false_score - true_score
        except Exception as e:
            logger.error(f"Error computing MCS: {str(e)}")
            return 0.0

    def select_samples(
        self,
        examples: List[Example]
    ) -> List[Sample]:
        """Select top-k samples based on MCS."""
        scored_samples = []
        
        for example in tqdm(examples, desc="Computing MCS scores"):
            try:
                mcs = self.compute_mcs(example.text, example.labels[0])
                scored_samples.append(
                    Sample(
                        text=example.text,
                        label=example.labels[0],
                        score=mcs
                    )
                )
            except Exception as e:
                logger.error(f"Error processing sample: {str(e)}")
                continue
        
        scored_samples.sort(
            key=lambda x: x.score if x.score is not None else float('-inf'),
            reverse=True
        )
        return scored_samples[:self.config.top_k_samples]

    def generate_prompt(
        self,
        samples: List[Sample],
        explanations: List[List[str]]
    ) -> str:
        """Generate the complete prompt with explanations."""
        prompt_parts = []
        
        for sample, explanation in zip(samples, explanations):
            # Format prompt with explanation
            rationale = (
                f"The key words: {', '.join(explanation[:self.config.top_k_keywords])} "
                f"are crucial clues for predicting {sample.label} as the correct answer."
            )
            
            prompt_part = (
                f"Input: {sample.text}\n"
                f"Rationale: {rationale}\n"
                f"Label: {sample.label}\n"
            )
            prompt_parts.append(prompt_part)
        
        return "\n".join(prompt_parts)

    def prepare_visualization_data(
        self,
        samples: List[Sample],
        explanations: List[np.ndarray]
    ) -> List[Dict]:
        """Prepare data for visualization."""
        visualization_data = []
        
        for sample, explanation in zip(samples, explanations):
            # Get tokens
            tokens = [
                clean_token(token) 
                for token in self.tokenizer.tokenize(sample.text)
            ]
            
            # Match tokens with scores
            token_scores = []
            for token, score in zip(tokens, explanation):
                if token and token.lower() not in stopwords.words('english'):
                    token_scores.append({
                        'token': token,
                        'score': float(score)
                    })
            
            # Normalize scores
            max_score = max(t['score'] for t in token_scores) if token_scores else 1
            for t in token_scores:
                t['score'] /= max_score
            
            visualization_data.append({
                'text': sample.text,
                'label': sample.label,
                'token_scores': token_scores
            })
        
        return visualization_data

    def process_examples(self, examples: List[Example]) -> ModelOutput:
        """Process examples through the AMPLIFY pipeline."""
        try:
            logger.info("Starting AMPLIFY pipeline")
            
            # Select samples
            logger.info("Selecting samples using MCS")
            selected_samples = self.select_samples(examples)
            
            # Generate explanations
            logger.info("Generating explanations")
            explanations = []
            explanation_vectors = []
            for sample in selected_samples:
                explanation = self.generate_explanation(
                    sample.text,
                    sample.label,
                    method="vanilla"
                )
                explanation_vectors.append(explanation)
                
                # Get top tokens for explanation
                tokens = [
                    clean_token(token)
                    for token in self.tokenizer.tokenize(sample.text)
                    if token.lower() not in stopwords.words('english')
                ]
                
                # Match tokens with explanation scores
                token_scores = list(zip(tokens, explanation))
                token_scores.sort(key=lambda x: x[1], reverse=True)
                top_tokens = [token for token, _ in token_scores[:self.config.top_k_keywords]]
                explanations.append(top_tokens)
                sample.explanation = top_tokens

            # Generate prompt
            logger.info("Generating final prompt")
            prompt = self.generate_prompt(selected_samples, explanations)
            
            # Prepare visualization data
            visualization_data = self.prepare_visualization_data(
                selected_samples,
                explanation_vectors
            )

            return ModelOutput(
                scores={
                    str(i): float(sample.score) if sample.score is not None else 0.0
                    for i, sample in enumerate(selected_samples)
                },
                explanations={
                    "token_explanations": explanations,
                    "explanation_vectors": explanation_vectors
                },
                visualizations={
                    "samples": visualization_data
                },
                additional_info={
                    "prompt": prompt,
                    "selected_samples": [
                        {
                            "text": sample.text,
                            "label": sample.label,
                            "score": sample.score,
                            "explanation": sample.explanation
                        }
                        for sample in selected_samples
                    ]
                }
            )

        except Exception as e:
            logger.error(f"Error in AMPLIFY pipeline: {str(e)}")
            raise

    def visualize_results(self, output: ModelOutput) -> None:
        """Visualize the results using the token visualization."""
        from .config import Visualizer
        Visualizer.create_amplify_token_visualization(
            output.visualizations["samples"]
        )