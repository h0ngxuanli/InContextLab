import torch
import logging
import plotly.graph_objects as go
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
from IPython.display import display, HTML
from jinja2 import Template
from .base import BaseICLModel, ModelRegistry, Example, ModelOutput

@dataclass
class ModelConfig:
    """Configuration for all models."""
    # General settings
    model_name: str = "gpt2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_train_steps: int = 1000
    output_dir: Optional[Path] = None
    random_seed: int = 42

    # AMPLIFY specific
    n_prefix_tokens: int = 1
    top_k_demons: int = 3
    top_k_keywords: int = 5
    gradient_method: str = "vanilla"
    
    # Latent Concept specific
    concept_learning_rate: float = 1e-4
    concept_train_steps: int = 500
    concept_batch_size: int = 16
    top_k_demons: int = 3
    
    # Information Flow specific
    num_layers: int = 12
    num_heads: int = 12
    flow_threshold: float = 1e-4
    normalize_flows: bool = True

    # Additional settings
    logging_steps: int = 100
    evaluation_steps: int = 200
    warmup_steps: int = 100
    weight_decay: float = 0.01
    additional_params: Dict[str, Any] = field(default_factory=dict)

class Visualizer:
    """Unified visualization utilities for all models."""
    
    @staticmethod
    def create_heatmap(data: np.ndarray, title: str) -> None:
        """Create interactive heatmap visualization."""
        fig = go.Figure(data=go.Heatmap(
            z=data,
            colorscale='Blues',
            showscale=True
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Attention Heads",
            yaxis_title="Layers",
            height=600,
            width=800,
            xaxis=dict(side="bottom"),
            yaxis=dict(autorange="reversed")
        )

        fig.show()

    @staticmethod
    def create_amplify_token_visualization(samples_with_scores: List[Dict]) -> None:
        """Create token visualization for AMPLIFY model."""
        html_template = '''
        <div style="font-family: Arial, sans-serif; max-width: 900px; margin: 20px auto;">
            <style>
                .token-container {
                    margin: 10px 0;
                    padding: 15px;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    background-color: #f8f9fa;
                }
                .token {
                    display: inline-block;
                    margin: 2px;
                    padding: 5px 10px;
                    border-radius: 4px;
                    position: relative;
                    cursor: pointer;
                    color: black;
                }
                .token:hover .tooltip {
                    display: block;
                }
                .tooltip {
                    display: none;
                    position: absolute;
                    bottom: 100%;
                    left: 50%;
                    transform: translateX(-50%);
                    background-color: #333;
                    color: white;
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 12px;
                    white-space: nowrap;
                    z-index: 100;
                }
                .prompt-container {
                    margin-top: 20px;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 8px;
                }
                .section-title {
                    font-weight: bold;
                    margin: 10px 0;
                    color: #2c5282;
                }
            </style>
            
            <div class="section-title">Token Explanation Score</div>
            
            <div id="visualization-container">
                {% for sample in samples %}
                <div class="token-container">
                    <div style="margin-bottom: 10px;">
                        <b>Sample {{ loop.index }}:</b> Label = {{ sample.label }} (MCS: {{ "%.3f"|format(sample.score) }})
                    </div>
                    <div>
                        {% for token in sample.token_scores %}
                        <span class="token" 
                              style="background-color: rgba(255, 69, 0, calc({{ token.score * 0.8 + 0.2 }}));">
                            {{ token.token }}
                            <span class="tooltip">Score: {{ "%.3f"|format(token.score) }}</span>
                        </span>
                        {% endfor %}
                    </div>
                </div>
                {% endfor %}
            </div>

            <div class="section-title">Generated Training Prompt</div>
            <div class="prompt-container">
                <pre style="margin: 0; white-space: pre-wrap;">{{ prompt }}</pre>
            </div>
        </div>
        '''

        # We no longer need to re-derive top tokens since they are in sample['explanation'].
        # Just normalize scores:
        for sample in samples_with_scores:
            scores = [t['score'] for t in sample['token_scores']]
            max_score = max(abs(min(scores)), abs(max(scores))) if scores else 1.0
            for token in sample['token_scores']:
                token['score'] = abs(token['score']) / max_score if max_score != 0 else 0.0

        # Use the explanation from the pipeline directly:
        prompt_parts = []
        for sample in samples_with_scores:
            # sample['explanation'] contains the top tokens chosen by the pipeline
            top_tokens = sample['explanation']
            prompt_part = (
                f"Input: {sample['text']}\n"
                f"Rationale: The key words: {', '.join(top_tokens)} are crucial clues "
                f"for predicting {sample['label']} as the correct answer.\n"
                f"Label: {sample['label']}"
            )
            prompt_parts.append(prompt_part)

        final_prompt = "\n\n".join(prompt_parts)

        template = Template(html_template)
        html_content = template.render(
            samples=samples_with_scores,
            prompt=final_prompt
        )
        
        display(HTML(html_content))

    def create_latent_concept_visualization(output: ModelOutput) -> None:
        """Create token visualization for Latent Concept model using ModelOutput."""
        demonstration_scores = []
        
        # Process each demonstration
        for demo in output.visualizations["demonstrations"]:
            demo_text = demo["demo"]["text"]
            token_list = output.explanations["token_contributions"][demo_text]
            
            # Find start and end indices between concept tokens
            start_idx = None
            end_idx = None
            for i, token in enumerate(token_list):
                if token["token"] == "<sentiment_concept>":
                    if start_idx is None:
                        start_idx = i + 1
                elif token["token"] == "<|endoftext|>":
                        end_idx = i
                        break
            
            # Filter to get only demonstration tokens
            filtered_contributions = token_list[start_idx:end_idx]
            
            # Normalize scores
            total_score = sum(contrib["score"] for contrib in filtered_contributions)
            for contrib in filtered_contributions:
                contrib["score"] /= total_score
                
            # Calculate scaled scores for visualization
            raw_scores = np.array([contrib["score"] for contrib in filtered_contributions])
            scaled_scores = (raw_scores**2) / (raw_scores**2).max()  # Quadratic scaling
            scaled_scores = 0.2 + 0.8 * scaled_scores  # Map to range [0.2, 1]
            
            for contrib, scaled_score in zip(filtered_contributions, scaled_scores):
                contrib["scaled_score"] = float(scaled_score)
                contrib["token"] = contrib["token"].lstrip('Ġ')
            
            demo_data = {
                "demo": {"text": demo_text},
                "score": demo["score"],
                "token_contributions": filtered_contributions
            }
            demonstration_scores.append(demo_data)

        html_template = '''
        <div style="font-family: Arial, sans-serif; max-width: 900px; margin: 20px auto;">
            <style>
                .section-title { font-weight: bold; margin: 10px 0; color: #2c5282; font-size: 20px; }
                .demonstration-card { margin: 10px 0; padding: 15px; border-radius: 8px; background-color: #f8f9fa; border: 1px solid #ddd; }
                .demonstration-header { font-size: 16px; font-weight: bold; color: #2c5282; margin-bottom: 10px; }
                .token-container { margin-top: 10px; display: flex; flex-wrap: wrap; }
                .token { display: inline-block; margin: 5px; padding: 5px 8px; border-radius: 4px; font-size: 14px; position: relative; cursor: pointer; background-color: rgb(255, 255, 255); }
                .token[data-score] { background-color: rgba(255, 69, 0, calc(var(--score))); color: black; }
                .token:hover { background-color: rgba(255, 0, 0, 1); }
                .token:hover .tooltip { display: block; }
                .tooltip { display: none; position: absolute; top: -30px; left: 50%; transform: translateX(-50%); background-color: #333; color: white; padding: 5px 8px; border-radius: 4px; font-size: 12px; z-index: 10; }
            </style>
            <div class="section-title">Selected Demonstrations and Token Contributions</div>
            <div id="demonstration-container">
                {% for demo in demonstration_scores %}
                <div class="demonstration-card">
                    <div class="demonstration-header">Demonstration Score: {{ "%.3f"|format(demo.score) }}</div>
                    <div>
                        <b>Token Contributions:</b>
                        <div class="token-container">
                            {% for token in demo.token_contributions %}
                            <div class="token" style="--score: {{ token.scaled_score }}" data-score="{{ token.scaled_score }}">
                                {{ token.token }}
                                <span class="tooltip">Score: {{ "%.2f"|format(token.score) }}</span>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        '''
        
        template = Template(html_template)
        html_content = template.render(demonstration_scores=demonstration_scores)
        display(HTML(html_content))


    @staticmethod
    def create_flow_visualization(
        layers: List[int],
        flow_metrics: Dict[str, List[float]],
        title: str
    ) -> None:
        """Create information flow visualization."""
        fig = go.Figure()

        for metric_name, values in flow_metrics.items():

            if metric_name == "S_wp":
                metric_name = "demon2label"
            elif metric_name == "S_pq":
                metric_name = "label2target"
            elif metric_name == "S_ww":
                metric_name = "demon2demon"

            fig.add_trace(go.Scatter(
                x=layers,
                y=values,
                mode='lines+markers',
                name=metric_name,
                marker=dict(size=8)
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Layer",
            yaxis_title="Flow Strength",
            height=500,
            width=800,
            legend=dict(
                x=0.1,
                y=1.1,
                orientation='h'
            ),
            margin=dict(l=50, r=50, t=80, b=50)
        )

        fig.show()

def setup_logging(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.setLevel(level)
        logger.propagate = False

    return logger