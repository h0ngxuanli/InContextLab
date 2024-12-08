import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import nltk
from dataclasses import dataclass
from .base import BaseICLModel, ModelRegistry, Example, ModelOutput
from .config import ModelConfig, setup_logging, Visualizer
import string

# Download required NLTK data (stopwords)
try:
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Warning: Failed to download stopwords: {e}")

logger = setup_logging(__name__)

# Define a more comprehensive set of meaningless tokens
stop_words = set(nltk.corpus.stopwords.words('english'))
extra_words = {"there", "this", "like", "am", "is", "he", "she", "/", "i"}
meaningless_tokens = stop_words.union(extra_words)
meaningless_tokens = meaningless_tokens.union(set(string.punctuation))

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
        self.setup_proxy_model()
        self.label_to_idx = {}  # Will be populated dynamically
        self.meaningless_tokens = meaningless_tokens

    def setup_proxy_model(self):
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

    def proxy_model_predict(self, text: str) -> Dict[str, float]:
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
        """Generate explanations using vanilla gradients for all tokens."""
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
            return token_attributions
            
        except Exception as e:
            logger.error(f"Error in vanilla gradients: {str(e)}")
            return np.zeros(1)

    def compute_contrastive_gradients(
        self,
        text: str,
        label: str
    ) -> np.ndarray:
        """Generate contrastive explanations for all tokens."""
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
            
            contrastive = true_gradients[0].cpu().numpy() - pred_gradients[0].cpu().numpy()
            token_contrast = np.linalg.norm(contrastive, ord=2, axis=-1)
            return token_contrast
            
        except Exception as e:
            logger.error(f"Error in contrastive gradients: {str(e)}")
            return np.zeros(1)

    def generate_explanation(
        self,
        text: str,
        label: str,
        method: str = "contrastive"
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
            target_idx = self.label_to_idx[true_label]
            predictions = self.proxy_model_predict(text)
            
            true_score = predictions.get(str(target_idx), 0)
            max_false_score = max(
                (score for label, score in predictions.items() if label != str(target_idx)),
                default=0
            )
            return max_false_score - true_score
        except Exception as e:
            logger.error(f"Error computing MCS: {str(e)}")
            return 0.0

    def select_demonstrations(
        self,
        demonstrations: List[Example]
    ) -> List[Sample]:
        """Select top-k samples based on negative MCS with largest absolute value."""
        scored_samples = []
        
        for example in tqdm(demonstrations, desc="Computing MCS scores"):
            try:
                mcs = self.compute_mcs(example.text, example.labels[0])
                # 只考虑MCS为负的样本
                if mcs < 0:
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
        
        # 对MCS为负的样本按MCS升序排序（即数值越小越排前）
        scored_samples.sort(
            key=lambda x: x.score if x.score is not None else float('inf'),
            reverse=False
        )
        
        return scored_samples[:self.config.top_k_demons]

    def generate_prompt(
        self,
        samples: List[Sample],
        explanations: List[List[str]]
    ) -> str:
        """Generate the complete prompt with explanations.

        Only meaningful tokens have been selected as top keywords, so no meaningless words appear here.
        """
        prompt_parts = []
        
        for sample, explanation in zip(samples, explanations):
            # explanation contains only meaningful tokens
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
        """Prepare data for visualization:
        - Show all tokens
        - Meaningful tokens get normalized scores (min-max)
        - Meaningless tokens have score = 0.0
        - Include demonstration score and explanation (top tokens) in the visualization
        """
        visualization_data = []

        for sample, explanation_vec in zip(samples, explanations):
            tokens = [clean_token(token) for token in self.tokenizer.tokenize(sample.text)]

            if not tokens or len(tokens) != len(explanation_vec):
                visualization_data.append({
                    'text': sample.text,
                    'label': sample.label,
                    'score': sample.score,
                    'explanation': sample.explanation,  # add the chosen explanation tokens
                    'token_scores': []
                })
                continue

            # Identify meaningful and meaningless tokens for normalization
            meaningful_pairs = [(t, s) for t, s in zip(tokens, explanation_vec)
                                if t.strip() != '' and t.lower() not in self.meaningless_tokens]

            if meaningful_pairs:
                scores = [s for _, s in meaningful_pairs]
                min_score = min(scores)
                max_score = max(scores)
            else:
                min_score = 0.0
                max_score = 0.0

            token_scores = []
            for token, score_val in zip(tokens, explanation_vec):
                token_lower = token.lower()
                if token_lower in self.meaningless_tokens or token.strip() == '':
                    norm_score = 0.0
                else:
                    if max_score > min_score:
                        norm_score = (score_val - min_score) / (max_score - min_score)
                    else:
                        norm_score = 0.5 if meaningful_pairs else 0.0

                token_scores.append({
                    'token': token,
                    'score': float(norm_score)
                })

            visualization_data.append({
                'text': sample.text,
                'label': sample.label,
                'score': sample.score,
                'explanation': sample.explanation,  # include final explanation tokens
                'token_scores': token_scores
            })

        return visualization_data

    def process_demonstrations(self, demonstrations: List[Example]) -> ModelOutput:
        """Process examples through the AMPLIFY pipeline."""
        try:
            logger.info("Starting AMPLIFY pipeline")
            
            # Dynamically build label-to-index mapping
            unique_labels = set()
            for example in demonstrations:
                for lab in example.labels:
                    unique_labels.add(lab)
            self.label_to_idx = {label: i for i, label in enumerate(sorted(unique_labels))}
            logger.info(f"Constructed label_to_idx mapping: {self.label_to_idx}")
            
            # Select samples
            logger.info("Selecting samples using MCS")
            selected_demonstrations = self.select_demonstrations(demonstrations)
            
            # Generate explanations
            logger.info("Generating explanations")
            explanations = []
            explanation_vectors = []
            for sample in selected_demonstrations:
                target_label_idx = self.label_to_idx[sample.label]
                # Compute full attributions (for all tokens)
                full_attributions = self.generate_explanation(
                    sample.text,
                    str(target_label_idx),
                    method=self.config.gradient_method
                )

                explanation_vectors.append(full_attributions)

                # Tokenize text
                tokens = [clean_token(token) for token in self.tokenizer.tokenize(sample.text)]
                # Consider only meaningful tokens for top keyword selection
                meaningful_tokens = [(tok, attr) for tok, attr in zip(tokens, full_attributions) 
                                     if tok.strip() != '' and tok.lower() not in self.meaningless_tokens]

                # Select top tokens from meaningful set
                meaningful_tokens.sort(key=lambda x: x[1], reverse=True)
                top_tokens = [token for token, _ in meaningful_tokens[:self.config.top_k_keywords]]

                explanations.append(top_tokens)
                sample.explanation = top_tokens

            # Generate prompt
            logger.info("Generating final prompt")
            prompt = self.generate_prompt(selected_demonstrations, explanations)
            
            # Prepare visualization data
            visualization_data = self.prepare_visualization_data(
                selected_demonstrations,
                explanation_vectors
            )

            return ModelOutput(
                scores={
                    str(i): float(sample.score) if sample.score is not None else 0.0
                    for i, sample in enumerate(selected_demonstrations)
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
                    "selected_demonstrations": [
                        {
                            "text": sample.text,
                            "label": sample.label,
                            "score": sample.score,
                            "explanation": sample.explanation
                        }
                        for sample in selected_demonstrations
                    ],
                    "label_to_idx": self.label_to_idx
                }
            )

        except Exception as e:
            logger.error(f"Error in AMPLIFY pipeline: {str(e)}")
            raise

    def visualize_results(self, output: ModelOutput) -> None:
        """Visualize the results using the token visualization."""
        Visualizer.create_amplify_token_visualization(
            output.visualizations["samples"]
        )