import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

class SemanticICLFramework:
    def __init__(self, model_name, dataset_name):
        """
        Initialize the framework with a model and dataset.

        Parameters:
            model_name (str): The Hugging Face model name.
            dataset_name (str): The dataset name (from Hugging Face).
        """
        self.model_name = model_name
        self.dataset_name = dataset_name

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load dataset
        self.dataset = load_dataset(dataset_name)

        # Preprocess dataset
        self.processed_data = self.preprocess_dataset()

    def preprocess_dataset(self):
        """
        Preprocess the dataset by tokenizing and extracting triplets.
        """
        processed = {}
        for split in self.dataset:
            processed[split] = [self.preprocess_example(example) for example in self.dataset[split]]
        return processed

    def preprocess_example(self, example):
        """
        Preprocess a single example by tokenizing text and extracting triplets.
        """
        text = self.extract_text(example)
        tokens = self.tokenizer(text, truncation=True, padding=True)
        triplets = self.extract_triplets(example)
        return {"text": text, "tokens": tokens, "triplets": triplets}

    def extract_text(self, example):
        """
        Extract text from the dataset example.
        """
        if self.dataset_name == "thunlp/few_rel":
            return " ".join(example.get("tokens", []))
        elif self.dataset_name == "Babelscape/rebel-dataset":
            return example.get("context", "")
        elif self.dataset_name == "thu-coai/kd_conv_with_kb":
            return example.get("content", "")
        else:
            raise ValueError("Unsupported dataset for text extraction.")

    def extract_triplets(self, example):
        """
        Extract triplets (head, relation, tail) from the example.
        """
        if self.dataset_name == "thunlp/few_rel":
            head = example.get("head", {}).get("text", "")
            tail = example.get("tail", {}).get("text", "")
            relation = example.get("relation", "")
            return [(head, relation, tail)]
        elif self.dataset_name == "Babelscape/rebel-dataset":
            return example.get("triplets", [])
        elif self.dataset_name == "thu-coai/kd_conv_with_kb":
            head = example.get("name", "")
            relation = example.get("attrname", "")
            tail = example.get("attrvalue", "")
            return [(head, relation, tail)]
        else:
            return []

    def analyze_attention_heads(self, input_ids, attention_mask):
        """
        Analyze attention heads for Query-Key and Output-Value circuits across various model architectures.
        """
        outputs = self.model(input_ids, attention_mask=attention_mask, output_attentions=True)
        attentions = outputs.attentions  # List of attention matrices from each layer
        return attentions

    def analyze_qk_ov_circuits(self, attentions):
        """
        Analyze QK and OV circuits to understand token relationships and output transformations.
        """
        qk_insights, ov_insights = {}, {}
        for layer_idx, attn_layer in enumerate(attentions):
            qk_matrix = np.mean(attn_layer[:, :, :, :], axis=0)  # Average over heads
            ov_matrix = np.mean(attn_layer[:, :, :, :], axis=0)  # Average over heads

            qk_insights[layer_idx] = qk_matrix
            ov_insights[layer_idx] = ov_matrix

        return {"QK": qk_insights, "OV": ov_insights}

    def layer_head_analysis(self, attentions):
        """
        Perform layer-wise and head-specific analysis of attention matrices.
        """
        layer_head_patterns = {}
        for layer_idx, layer_attn in enumerate(attentions):
            head_patterns = []
            for head_idx in range(layer_attn.shape[1]):  # Iterate over attention heads
                avg_attn = layer_attn[:, head_idx, :, :].mean(axis=0)
                head_patterns.append(avg_attn)

            layer_head_patterns[layer_idx] = head_patterns
        return layer_head_patterns

    def attribute_attention(self, attentions, triplets, input_ids):
        """
        Attribute attention scores to triplet components.
        """
        attribution_scores = []
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        for triplet in triplets:
            head, relation, tail = triplet
            head_idx = tokens.index(head) if head in tokens else None
            tail_idx = tokens.index(tail) if tail in tokens else None

            if head_idx is not None and tail_idx is not None:
                avg_attn = np.mean([attn[:, head_idx, tail_idx].mean() for attn in attentions])
                attribution_scores.append((triplet, avg_attn))

        return attribution_scores

    def visualize_layer_dynamics(self, layer_head_patterns):
        """
        Visualize attention dynamics across layers and heads.
        """
        for layer_idx, heads in layer_head_patterns.items():
            plt.figure(figsize=(10, 6))
            for head_idx, head_attn in enumerate(heads):
                plt.plot(head_attn.mean(axis=0), label=f"Head {head_idx}")
            plt.title(f"Layer {layer_idx} Attention Dynamics")
            plt.xlabel("Tokens")
            plt.ylabel("Attention Score")
            plt.legend()
            plt.show()

    def correlate_attention_with_metrics(self, relation_indices, icl_metrics):
        """
        Compute and visualize correlation between attention and ICL metrics.
        """
        correlation = np.corrcoef(relation_indices, icl_metrics["loss_reduction"])[0, 1]

        plt.figure(figsize=(8, 6))
        plt.scatter(relation_indices, icl_metrics["loss_reduction"], alpha=0.7)
        plt.title("Correlation Between Attention and Loss Reduction")
        plt.xlabel("Relation Index")
        plt.ylabel("Loss Reduction")
        plt.grid(True)
        plt.show()

        return correlation


# Example Usage
framework = SemanticICLFramework(model_name="gpt2", dataset_name="Babelscape/rebel-dataset")
processed_data = framework.processed_data["train"]

relation_indices = []
for example in processed_data[:5]:
    tokens = framework.tokenizer(example["text"], return_tensors="pt", truncation=True, padding=True)
    input_ids, attention_mask = tokens["input_ids"], tokens["attention_mask"]
    attentions = framework.analyze_attention_heads(input_ids, attention_mask)

    # Visualize attention for the first example
    framework.visualize_layer_dynamics(framework.layer_head_analysis(attentions))

    # Compute triplet attribution
    attribution = framework.attribute_attention(attentions, example["triplets"], input_ids)
    print(f"Attention Attribution: {attribution}")

# Example Correlation Visualization
icl_metrics = {"loss_reduction": [0.1, 0.2, 0.15, 0.18, 0.12]}
correlation = framework.correlate_attention_with_metrics(relation_indices, icl_metrics)
print(f"Correlation: {correlation}")
