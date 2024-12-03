# icl/core/layer_analysis.py

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer


class SaliencyAnalyzer:
    """Analyzes saliency scores for information flow in transformer layers."""

    def __init__(self, model: PreTrainedModel):
        self.model = model

    def compute_saliency_scores(
        self,
        attention_matrices: List[torch.Tensor],
        loss: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Computes saliency scores for attention matrices.

        Args:
            attention_matrices: List of attention matrices per layer.
            loss: Task-specific loss.

        Returns:
            List of tensors containing saliency scores per layer.
        """
        # Ensure attention matrices retain gradients
        for A in attention_matrices:
            A.retain_grad()

        # Backward pass to compute gradients
        loss.backward(retain_graph=True)

        # Compute saliency scores
        saliency_scores = []
        for A in attention_matrices:
            grad = A.grad  # Gradient w.r.t. attention matrix
            saliency = torch.abs(A * grad)
            saliency_scores.append(saliency)

        return saliency_scores

    def compute_flow_metrics(
        self,
        saliency_scores: List[torch.Tensor],
        class_positions: torch.Tensor,
        target_position: int,
    ) -> Dict[str, List[float]]:
        """
        Computes text-to-label and label-to-target flow metrics.

        Args:
            saliency_scores: List of saliency score tensors per layer.
            class_positions: Positions of label words.
            target_position: Position of target token.

        Returns:
            Dictionary containing S_wp, S_pq, and S_ww scores per layer.
        """
        S_wp_list = []
        S_pq_list = []
        S_ww_list = []

        for saliency in saliency_scores:
            # saliency shape: [batch_size, num_heads, seq_len, seq_len]
            S_wp = self._compute_average_flow(
                saliency, self._get_text_to_label_connections(class_positions)
            )
            S_pq = self._compute_average_flow(
                saliency, self._get_label_to_target_connections(class_positions, target_position)
            )
            S_ww = self._compute_average_flow(
                saliency,
                self._get_other_connections(
                    class_positions, target_position, saliency.shape[-1]
                ),
            )

            S_wp_list.append(S_wp)
            S_pq_list.append(S_pq)
            S_ww_list.append(S_ww)

        return {"S_wp": S_wp_list, "S_pq": S_pq_list, "S_ww": S_ww_list}

    def _get_text_to_label_connections(
        self, class_positions: torch.Tensor
    ) -> List[Tuple[int, int]]:
        """Gets connections from input text to label words."""
        connections = []
        for pos in class_positions.tolist():
            connections.extend([(pos, j) for j in range(pos)])
        return connections

    def _get_label_to_target_connections(
        self, class_positions: torch.Tensor, target_position: int
    ) -> List[Tuple[int, int]]:
        """Gets connections from label words to target position."""
        return [(target_position, pos) for pos in class_positions.tolist()]

    def _get_other_connections(
        self, class_positions: torch.Tensor, target_position: int, seq_len: int
    ) -> List[Tuple[int, int]]:
        """Gets all other connections excluding those involving label words and target."""
        all_connections = [(i, j) for i in range(seq_len) for j in range(seq_len) if j < i]
        exclude_connections = set(
            self._get_text_to_label_connections(class_positions)
            + self._get_label_to_target_connections(class_positions, target_position)
        )
        other_connections = list(set(all_connections) - exclude_connections)
        return other_connections

    def _compute_average_flow(
        self, saliency_scores: torch.Tensor, connections: List[Tuple[int, int]]
    ) -> float:
        """Computes average saliency score for given connections."""
        if not connections:
            return 0.0
        total_saliency = 0.0
        num_heads = saliency_scores.shape[1]  # [batch_size, num_heads, seq_len, seq_len]
        for i, j in connections:
            total_saliency += saliency_scores[:, :, i, j].sum().item()
        average_saliency = total_saliency / (len(connections) * num_heads)
        return average_saliency


class InformationBlocker:
    """Implements attention blocking mechanisms for ablation studies."""

    def block_label_connections(
        self,
        attention_matrices: List[torch.Tensor],
        label_positions: List[int],
        layers_to_block: List[int],
    ) -> List[torch.Tensor]:
        """
        Blocks attention connections to/from label words in specified layers.

        Args:
            attention_matrices: List of attention matrices per layer.
            label_positions: Positions of label words to block.
            layers_to_block: Indices of layers to apply blocking.

        Returns:
            List of modified attention matrices.
        """
        blocked_matrices = attention_matrices.copy()
        for layer_idx in layers_to_block:
            A = attention_matrices[layer_idx].clone()  # [batch_size, num_heads, seq_len, seq_len]
            for pos in label_positions:
                # Block connections from positions before 'pos' to 'pos'
                A[:, :, pos, :pos] = 0
            blocked_matrices[layer_idx] = A
        return blocked_matrices

    def compute_loyalty_metrics(
        self, original_outputs: torch.Tensor, blocked_outputs: torch.Tensor
    ) -> Dict[str, float]:
        """
        Computes loyalty metrics comparing original and blocked outputs.

        Args:
            original_outputs: Model outputs without blocking (logits).
            blocked_outputs: Model outputs with attention blocking (logits).

        Returns:
            Dictionary with label and word loyalty scores.
        """
        # Label loyalty - consistency of output labels
        original_labels = original_outputs.argmax(dim=-1)
        blocked_labels = blocked_outputs.argmax(dim=-1)
        label_loyalty = (original_labels == blocked_labels).float().mean().item()

        # Word loyalty - Jaccard similarity of top-5 predictions
        orig_top5 = torch.topk(original_outputs, k=5, dim=-1).indices
        block_top5 = torch.topk(blocked_outputs, k=5, dim=-1).indices

        word_loyalty = self._compute_jaccard_similarity(orig_top5, block_top5)

        return {"label_loyalty": label_loyalty, "word_loyalty": word_loyalty}

    def _compute_jaccard_similarity(self, set1: torch.Tensor, set2: torch.Tensor) -> float:
        """Computes Jaccard similarity between two sets of token indices."""
        set1 = set(set1.flatten().tolist())
        set2 = set(set2.flatten().tolist())
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0


class AttentionModeler:
    """Models and analyzes attention mechanisms."""

    def compute_attention_weights(
        self, query: torch.Tensor, key: torch.Tensor, dim: int
    ) -> torch.Tensor:
        """
        Computes attention weights using query and key vectors.

        Args:
            query: Query tensor of shape [batch_size, num_heads, seq_len, head_dim].
            key: Key tensor of shape [batch_size, num_heads, seq_len, head_dim].
            dim: Scaling dimension (usually head_dim).

        Returns:
            Attention weight tensor of shape [batch_size, num_heads, seq_len, seq_len].
        """
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / np.sqrt(dim)
        return torch.softmax(scores, dim=-1)

    def fit_logistic_regression(
        self, attention_weights: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Fits logistic regression model to analyze attention patterns.

        Args:
            attention_weights: Attention weights tensor of shape [batch_size, num_heads, seq_len, seq_len].
            labels: True labels (tensor of shape [batch_size]).

        Returns:
            Dictionary with regression coefficients and metrics.
        """
        # Flatten attention weights
        x = attention_weights.view(attention_weights.size(0), -1)
        num_classes = labels.max().item() + 1

        # Convert labels to tensor
        y = labels

        # Define linear model
        model = torch.nn.Linear(x.size(1), num_classes)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Training loop
        epochs = 100
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # Extract coefficients
        coefficients = model.weight.detach()
        return {"coefficients": coefficients, "num_classes": num_classes}


class ContextCompressor:
    """Implements context compression techniques."""

    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.stored_states = {}

    def compress_demonstrations(
        self, demonstrations: List[Dict], label_positions: List[int]
    ) -> torch.Tensor:
        """
        Compresses demonstrations into pre-computed hidden states.

        Args:
            demonstrations: List of demonstration examples.
            label_positions: Positions of label words.

        Returns:
            Compressed hidden states tensor.
        """
        compressed_states = []

        for demo in demonstrations:
            # Get hidden states for demonstration
            with torch.no_grad():
                outputs = self.model(**demo, output_hidden_states=True, return_dict=True)
            states = outputs.hidden_states  # Tuple of (layer_num + 1) tensors
            # Extract final hidden state (last layer)
            final_hidden_state = states[-1]  # [batch_size, seq_len, hidden_size]
            # Extract states at label positions
            label_states = final_hidden_state[:, label_positions, :]  # [batch_size, num_labels, hidden_size]
            compressed_states.append(label_states)

        # Concatenate compressed states along sequence dimension
        compressed_states = torch.cat(compressed_states, dim=1)  # [batch_size, total_labels, hidden_size]
        return compressed_states

    def analyze_compression_error(
        self, original_outputs: torch.Tensor, compressed_outputs: torch.Tensor
    ) -> Dict[str, float]:
        """
        Analyzes errors introduced by compression.

        Args:
            original_outputs: Model outputs with full context (logits).
            compressed_outputs: Outputs with compressed context (logits).

        Returns:
            Dictionary with error metrics.
        """
        # Compute Mean Squared Error
        mse = torch.mean((original_outputs - compressed_outputs) ** 2).item()
        # Compute Cosine Similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            original_outputs.view(-1), compressed_outputs.view(-1), dim=0
        ).item()

        return {"mse": mse, "cosine_similarity": cos_sim}


class ErrorAnalyzer:
    """Analyzes classification errors and confusion patterns."""

    def compute_confusion_matrix(
        self, key_vectors: torch.Tensor, predictions: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes actual and predicted confusion matrices.

        Args:
            key_vectors: Key vectors for label words (shape: [num_classes, hidden_size]).
            predictions: Model predictions (shape: [batch_size]).
            labels: True labels (shape: [batch_size]).

        Returns:
            Tuple of (predicted_confusion, actual_confusion) matrices.
        """
        num_classes = key_vectors.size(0)

        # Compute predicted confusion based on key vector distances
        key_vectors_norm = key_vectors / key_vectors.norm(dim=1, keepdim=True)
        dist_matrix = 1 - torch.mm(key_vectors_norm, key_vectors_norm.t())  # Cosine distance

        max_dist = dist_matrix.max()
        pred_confusion = dist_matrix / max_dist  # Normalize distances

        # Compute actual confusion matrix
        actual_confusion = torch.zeros((num_classes, num_classes))
        for pred, true in zip(predictions, labels):
            actual_confusion[true.item(), pred.item()] += 1

        # Normalize actual confusion
        if actual_confusion.sum() > 0:
            actual_confusion = actual_confusion / actual_confusion.sum()

        return pred_confusion, actual_confusion

    def analyze_error_patterns(
        self, confusion_matrices: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Analyzes patterns in confusion matrices.

        Args:
            confusion_matrices: Tuple of (predicted_confusion, actual_confusion) matrices.

        Returns:
            Dictionary with error analysis metrics.
        """
        pred_conf, actual_conf = confusion_matrices

        # Flatten and remove diagonal elements
        pred_conf_flat = pred_conf.flatten()
        actual_conf_flat = actual_conf.flatten()
        mask = ~torch.eye(pred_conf.size(0)).bool().flatten()
        pred_conf_flat = pred_conf_flat[mask]
        actual_conf_flat = actual_conf_flat[mask]

        # Compute correlation coefficient
        correlation_matrix = torch.corrcoef(torch.stack([pred_conf_flat, actual_conf_flat]))
        correlation = correlation_matrix[0, 1].item() if correlation_matrix.numel() > 1 else 0.0

        # Get top confusion pairs
        top_confusions = torch.topk(actual_conf_flat, k=5)

        return {
            "prediction_correlation": correlation,
            "top_confusion_pairs": top_confusions.indices.tolist(),
            "top_confusion_values": top_confusions.values.tolist(),
        }


class ICLVisualizer:
    """Visualization utilities for ICL analysis results."""

    @staticmethod
    def plot_information_flow(
        layers: List[int],
        Swp: List[float],
        Spq: List[float],
        Sww: List[float],
        title: str = "",
        save_path: Optional[str] = None,
    ):
        """
        Plots information flow metrics across layers.

        Args:
            layers: Layer indices.
            Swp: Text-to-label word flow values.
            Spq: Label-to-target flow values.
            Sww: Other information flow values.
            title: Plot title.
            save_path: Path to save figure.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(layers, Swp, label='S_wp', color='blue')
        plt.plot(layers, Spq, label='S_pq', color='orange')
        plt.plot(layers, Sww, label='S_ww', color='green')

        plt.xlabel('Layer')
        plt.ylabel('S')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_isolation_impact(metrics: Dict[str, List[float]], title: str = "", save_path: Optional[str] = None):
        """
        Plots impact of isolation experiments.

        Args:
            metrics: Dictionary containing different conditions and their scores.
            title: Plot title.
            save_path: Path to save figure.
        """
        categories = ['Label Loyalty\n(GPT2-XL)', 'Word Loyalty\n(GPT2-XL)',
                      'Label Loyalty\n(GPT-J)', 'Word Loyalty\n(GPT-J)']

        x = np.arange(len(categories))
        width = 0.15

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot bars for each condition
        ax.bar(x - 2*width, metrics['no_isolation'], width, label='No Isolation')
        ax.bar(x - width, metrics['label_first'], width, label='Label Words (First)')
        ax.bar(x, metrics['label_last'], width, label='Label Words (Last)')
        ax.bar(x + width, metrics['random_first'], width, label='Random (First)')
        ax.bar(x + 2*width, metrics['random_last'], width, label='Random (Last)')

        ax.set_ylabel('Loyalty')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_confusion_matrix(
        matrix: np.ndarray,
        labels: List[str],
        title: str = "",
        save_path: Optional[str] = None,
    ):
        """
        Plots confusion matrix heatmap.

        Args:
            matrix: Confusion matrix values.
            labels: Category labels.
            title: Plot title.
            save_path: Path to save figure.
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, cmap='YlOrRd')

        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center')

        plt.colorbar()
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.yticks(range(len(labels)), labels)
        plt.title(title)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()


class ICLResultsGenerator:
    """Orchestrates ICL analysis and generates results using component classes."""

    def __init__(self, model: PreTrainedModel, tokenizer: AutoTokenizer, predictor):
        self.model = model
        self.tokenizer = tokenizer
        self.predictor = predictor

        # Initialize analysis components
        self.saliency_analyzer = SaliencyAnalyzer(model)
        self.info_blocker = InformationBlocker()
        self.attention_modeler = AttentionModeler()
        self.context_compressor = ContextCompressor(model)
        self.error_analyzer = ErrorAnalyzer()

        # Initialize visualization helper
        self.visualizer = ICLVisualizer()

    def generate_information_flow_analysis(
        self, dataset: Dict, n_demos: int = 1, n_samples: int = 100
    ) -> Dict:
        """
        Generates complete information flow analysis (Section 2.1-2.3).

        Args:
            dataset: Dataset containing demos and test samples.
            n_demos: Number of demonstrations per class.
            n_samples: Number of test samples to analyze.

        Returns:
            Dictionary containing analysis results.
        """
        results = {}

        # 1. Compute saliency metrics (Section 2.1)
        print("Computing saliency metrics...")
        demos, samples = self._prepare_data(dataset, n_demos, n_samples)

        inputs = self._prepare_inputs(demos, samples)
        inputs['labels'] = samples['labels']

        # Get model outputs with attention
        outputs = self.model(**inputs, output_attentions=True, return_dict=True)
        attention_matrices = outputs.attentions  # List of attention matrices per layer

        # Compute loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), inputs['labels'].view(-1))

        # Compute saliency scores
        saliency_scores = self.saliency_analyzer.compute_saliency_scores(
            attention_matrices=attention_matrices, loss=loss
        )

        # Get class and target positions
        class_positions = self.predictor.get_label_positions(inputs)
        target_position = self.predictor.get_target_position(inputs)

        flow_metrics = self.saliency_analyzer.compute_flow_metrics(
            saliency_scores=saliency_scores,
            class_positions=class_positions,
            target_position=target_position,
        )

        results['saliency'] = flow_metrics

        # Additional analysis steps (blocking, deep analysis, error analysis)
        # ...

        return results

    def visualize_results(self, results: Dict, save_dir: Optional[str] = None):
        """
        Generates all visualizations from the paper.

        Args:
            results: Results dictionary from generate_information_flow_analysis.
            save_dir: Optional directory to save figures.
        """
        # 1. Plot information flow (Figure 3)
        self.visualizer.plot_information_flow(
            layers=list(range(len(results['saliency']['S_wp']))),
            Swp=results['saliency']['S_wp'],
            Spq=results['saliency']['S_pq'],
            Sww=results['saliency']['S_ww'],
            title="Information Flow Analysis",
            save_path=f"{save_dir}/figure3.png" if save_dir else None,
        )

        # Additional visualization steps
        # ...

    def _prepare_data(
        self, dataset: Dict, n_demos: int, n_samples: int
    ) -> Tuple[Dict, Dict]:
        """Prepares demonstration and test data."""
        demos = self._sample_demonstrations(dataset, n_demos)
        samples = self._sample_test_data(dataset, n_samples)
        return demos, samples

    def _prepare_inputs(self, demos: Dict, samples: Dict) -> Dict:
        """Prepares model inputs by combining demos and samples."""
        # Tokenize and combine demos and samples
        # Implementation depends on dataset and tokenizer
        pass

    def _get_key_vectors(self, samples: Dict) -> torch.Tensor:
        """Gets key vectors for label positions."""
        # Implementation depends on model architecture
        pass

    def _get_attention_weights(self, samples: Dict) -> torch.Tensor:
        """Gets attention weights for target position."""
        # Implementation depends on model architecture
        pass

    @staticmethod
    def _sample_demonstrations(dataset: Dict, n_demos: int) -> Dict:
        """Samples demonstrations from dataset."""
        # Implementation for sampling demonstrations
        pass

    @staticmethod
    def _sample_test_data(dataset: Dict, n_samples: int) -> Dict:
        """Samples test data from dataset."""
        # Implementation for sampling test data
        pass








# import torch
# import numpy as np
# from typing import List, Dict, Tuple, Optional
# import torch
# from transformers import PreTrainedModel
# import matplotlib.pyplot as plt
# import numpy as np
        
 
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import matplotlib.pyplot as plt
# import numpy as np
# from typing import List, Dict, Tuple


# class SaliencyAnalyzer:
#     """Analyzes saliency scores for information flow in transformer layers."""
    
#     def __init__(self, model):
#         self.model = model
    
#     def compute_saliency_scores(self, attention_matrices: List[torch.Tensor], 
#                               loss: torch.Tensor) -> torch.Tensor:
#         """
#         Computes saliency scores for attention matrices.
        
#         Args:
#             attention_matrices: List of attention matrices per layer
#             loss: Task-specific loss
            
#         Returns:
#             Tensor of saliency scores
#         """
#         I_l = torch.zeros_like(attention_matrices[0])
        
#         for A_h in attention_matrices:
#             grads = torch.autograd.grad(loss, A_h, retain_graph=True)[0]
#             I_l += torch.abs(A_h * grads)
            
#         return I_l

#     def compute_flow_metrics(self, saliency_scores: torch.Tensor,
#                            class_positions: torch.Tensor,
#                            target_position: torch.Tensor) -> Dict[str, float]:
#         """
#         Computes text-to-label and label-to-target flow metrics.
        
#         Args:
#             saliency_scores: Computed saliency scores
#             class_positions: Positions of label words
#             target_position: Position of target token
            
#         Returns:
#             Dictionary containing S_wp, S_pq and S_ww scores
#         """
#         # Text to label words flow (S_wp)
#         C_wp = self._get_text_to_label_connections(class_positions)
#         S_wp = self._compute_average_flow(saliency_scores, C_wp)
        
#         # Label words to target flow (S_pq)
#         C_pq = self._get_label_to_target_connections(class_positions, target_position)
#         S_pq = self._compute_average_flow(saliency_scores, C_pq)
        
#         # Other information flow (S_ww)
#         C_ww = self._get_other_connections(class_positions, target_position)
#         S_ww = self._compute_average_flow(saliency_scores, C_ww)
        
#         return {
#             'S_wp': S_wp,
#             'S_pq': S_pq,
#             'S_ww': S_ww
#         }

#     def _get_text_to_label_connections(self, class_positions: torch.Tensor) -> List[Tuple[int, int]]:
#         """Gets connections from input text to label words."""
#         connections = []
#         for pos in class_positions:
#             connections.extend([(pos, j) for j in range(pos)])
#         return connections
    
#     def _get_label_to_target_connections(self, class_positions: torch.Tensor, 
#                                        target_position: int) -> List[Tuple[int, int]]:
#         """Gets connections from label words to target position."""
#         return [(target_position, pos) for pos in class_positions]
    
#     def _compute_average_flow(self, saliency_scores: torch.Tensor, 
#                             connections: List[Tuple[int, int]]) -> float:
#         """Computes average saliency score for given connections."""
#         if not connections:
#             return 0.0
#         total = sum(saliency_scores[i,j] for i,j in connections)
#         return total / len(connections)

# class InformationBlocker:
#     """Implements attention blocking mechanisms for ablation studies."""
    
#     def __init__(self):
#         self.block_masks = {}
    
#     def block_label_connections(self, attention_matrix: torch.Tensor,
#                               label_positions: List[int]) -> torch.Tensor:
#         """
#         Blocks attention connections to/from label words.
        
#         Args:
#             attention_matrix: Original attention matrix
#             label_positions: Positions of label words to block
            
#         Returns:
#             Modified attention matrix with blocked connections
#         """
#         blocked_matrix = attention_matrix.clone()
#         for pos in label_positions:
#             blocked_matrix[:, :, pos, :pos] = 0
#         return blocked_matrix
    
#     def compute_loyalty_metrics(self, original_outputs: torch.Tensor,
#                               blocked_outputs: torch.Tensor) -> Dict[str, float]:
#         """
#         Computes loyalty metrics comparing original and blocked outputs.
        
#         Args:
#             original_outputs: Model outputs without blocking
#             blocked_outputs: Model outputs with attention blocking
            
#         Returns:
#             Dictionary with label and word loyalty scores
#         """
#         # Label loyalty - consistency of output labels
#         label_loyalty = (original_outputs.argmax(-1) == 
#                         blocked_outputs.argmax(-1)).float().mean()
        
#         # Word loyalty - Jaccard similarity of top-5 predictions
#         orig_top5 = torch.topk(original_outputs, k=5, dim=-1).indices
#         block_top5 = torch.topk(blocked_outputs, k=5, dim=-1).indices
        
#         word_loyalty = self._compute_jaccard_similarity(orig_top5, block_top5)
        
#         return {
#             'label_loyalty': label_loyalty,
#             'word_loyalty': word_loyalty
#         }
    
#     def _compute_jaccard_similarity(self, set1: torch.Tensor, 
#                                   set2: torch.Tensor) -> float:
#         """Computes Jaccard similarity between two sets of token indices."""
#         intersection = len(set(set1.flatten().tolist()) & 
#                          set(set2.flatten().tolist()))
#         union = len(set(set1.flatten().tolist()) | 
#                    set(set2.flatten().tolist()))
#         return intersection / union if union > 0 else 0.0

# class AttentionModeler:
#     """Models and analyzes attention mechanisms."""
    
#     def compute_attention_weights(self, query: torch.Tensor,
#                                 key: torch.Tensor,
#                                 dim: int) -> torch.Tensor:
#         """
#         Computes attention weights using query and key vectors.
        
#         Args:
#             query: Query vector
#             key: Key vector
#             dim: Dimension for scaling
            
#         Returns:
#             Attention weight matrix
#         """
#         scores = torch.matmul(query, key.transpose(-2, -1))
#         scores = scores / np.sqrt(dim)
#         return torch.softmax(scores, dim=-1)
    
#     def fit_logistic_regression(self, attention_weights: torch.Tensor,
#                               labels: torch.Tensor) -> Dict[str, torch.Tensor]:
#         """
#         Fits logistic regression model to analyze attention patterns.
        
#         Args:
#             attention_weights: Computed attention weights
#             labels: True labels
            
#         Returns:
#             Dictionary with regression coefficients and metrics
#         """
#         # Implement logistic regression fitting
#         # This is a simplified version - you may want to use sklearn or similar
#         x = attention_weights.view(attention_weights.size(0), -1)
#         num_classes = labels.max() + 1
        
#         betas = []
#         for i in range(num_classes):
#             y_i = (labels == i).float()
#             beta_i = self._fit_single_class(x, y_i)
#             betas.append(beta_i)
            
#         return {
#             'coefficients': torch.stack(betas),
#             'num_classes': num_classes
#         }
    
#     def _fit_single_class(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         """Fits logistic regression for a single class."""
#         # Simplified implementation - in practice use proper optimization
#         beta = torch.zeros(x.size(1))
#         lr = 0.01
#         epochs = 100
        
#         for _ in range(epochs):
#             y_pred = torch.sigmoid(x @ beta)
#             grad = x.t() @ (y_pred - y)
#             beta = beta - lr * grad
            
#         return beta

# class ContextCompressor:
#     """Implements context compression techniques."""
    
#     def __init__(self, model):
#         self.model = model
#         self.stored_states = {}
        
#     def compress_demonstrations(self, 
#                               demonstrations: List[Dict],
#                               label_positions: List[int]) -> torch.Tensor:
#         """
#         Compresses demonstrations into pre-computed hidden states.
        
#         Args:
#             demonstrations: List of demonstration examples
#             label_positions: Positions of label words
            
#         Returns:
#             Compressed hidden states
#         """
#         hidden_states = []
        
#         for demo in demonstrations:
#             # Get hidden states for demonstration
#             outputs = self.model(**demo)
#             states = outputs.hidden_states
            
#             # Extract states at label positions
#             label_states = torch.stack([
#                 states[i][:, pos] 
#                 for i, pos in enumerate(label_positions)
#             ])
            
#             hidden_states.append(label_states)
            
#         return torch.stack(hidden_states)
    
#     def analyze_compression_error(self, 
#                                 original_outputs: torch.Tensor,
#                                 compressed_outputs: torch.Tensor) -> Dict[str, float]:
#         """
#         Analyzes errors introduced by compression.
        
#         Args:
#             original_outputs: Model outputs with full context
#             compressed_outputs: Outputs with compressed context
            
#         Returns:
#             Dictionary with error metrics
#         """
#         # Compute various error metrics
#         mse = torch.mean((original_outputs - compressed_outputs) ** 2)
#         cos_sim = torch.nn.functional.cosine_similarity(
#             original_outputs.view(-1),
#             compressed_outputs.view(-1)
#         )
        
#         return {
#             'mse': mse.item(),
#             'cosine_similarity': cos_sim.item()
#         }

# class ErrorAnalyzer:
#     """Analyzes classification errors and confusion patterns."""
    
#     def compute_confusion_matrix(self, 
#                                key_vectors: torch.Tensor,
#                                predictions: torch.Tensor,
#                                labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Computes actual and predicted confusion matrices.
        
#         Args:
#             key_vectors: Key vectors for label words
#             predictions: Model predictions
#             labels: True labels
            
#         Returns:
#             Tuple of (predicted_confusion, actual_confusion)
#         """
#         num_classes = len(key_vectors)
        
#         # Compute predicted confusion based on key vector distances
#         pred_confusion = torch.zeros((num_classes, num_classes))
#         for i in range(num_classes):
#             for j in range(num_classes):
#                 if i != j:
#                     dist = torch.norm(key_vectors[i] - key_vectors[j])
#                     max_dist = torch.max(torch.norm(key_vectors.unsqueeze(1) - 
#                                                   key_vectors.unsqueeze(0)))
#                     pred_confusion[i,j] = dist / max_dist
                    
#         # Compute actual confusion matrix
#         actual_confusion = torch.zeros((num_classes, num_classes))
#         for pred, true in zip(predictions, labels):
#             if pred != true:
#                 actual_confusion[true, pred] += 1
                
#         # Normalize actual confusion
#         actual_confusion = actual_confusion / actual_confusion.sum()
        
#         return pred_confusion, actual_confusion
    
#     def analyze_error_patterns(self,
#                              confusion_matrices: Tuple[torch.Tensor, torch.Tensor]
#                              ) -> Dict[str, float]:
#         """
#         Analyzes patterns in confusion matrices.
        
#         Args:
#             confusion_matrices: (predicted_confusion, actual_confusion) matrices
            
#         Returns:
#             Dictionary with error analysis metrics
#         """
#         pred_conf, actual_conf = confusion_matrices
        
#         correlation = torch.corrcoef(
#             torch.stack([pred_conf.view(-1), actual_conf.view(-1)])
#         )[0,1]
        
#         top_confusions = torch.topk(actual_conf.view(-1), k=5)
        
#         return {
#             'prediction_correlation': correlation.item(),
#             'top_confusion_pairs': top_confusions.indices.tolist(),
#             'top_confusion_values': top_confusions.values.tolist()
#         }
        


# class ICLVisualizer:
#     """Visualization utilities for ICL analysis results."""
    
#     @staticmethod
#     def plot_information_flow(layers: list, Swp: list, Spq: list, Sww: list, 
#                             title: str = "", save_path: str = None):
#         """
#         Plots information flow metrics across layers (reproduces Figure 3).
        
#         Args:
#             layers: Layer indices
#             Swp: Text-to-label word flow values
#             Spq: Label-to-target flow values
#             Sww: Other information flow values
#             title: Plot title
#             save_path: Path to save figure
#         """
#         plt.figure(figsize=(10, 6))
#         plt.plot(layers, Swp, label='Swp', color='blue')
#         plt.plot(layers, Spq, label='Spq', color='orange')
#         plt.plot(layers, Sww, label='Sww', color='green')
        
#         plt.xlabel('Layer')
#         plt.ylabel('S')
#         plt.title(title)
#         plt.legend()
#         plt.grid(True, alpha=0.3)
        
#         if save_path:
#             plt.savefig(save_path)
#         plt.show()

#     @staticmethod
#     def plot_isolation_impact(metrics: dict, title: str = "", save_path: str = None):
#         """
#         Plots impact of isolation experiments (reproduces Figure 4).
        
#         Args:
#             metrics: Dictionary containing:
#                 - 'no_isolation': Baseline scores
#                 - 'label_first': Scores with label word isolation in first layers
#                 - 'label_last': Scores with label word isolation in last layers
#                 - 'random_first': Scores with random word isolation in first layers
#                 - 'random_last': Scores with random word isolation in last layers
#             title: Plot title
#             save_path: Path to save figure
#         """
#         categories = ['Label Loyalty\n(GPT2-XL)', 'Word Loyalty\n(GPT2-XL)',
#                      'Label Loyalty\n(GPT-J)', 'Word Loyalty\n(GPT-J)']
        
#         x = np.arange(len(categories))
#         width = 0.15
        
#         fig, ax = plt.subplots(figsize=(12, 6))
        
#         # Plot bars for each condition
#         ax.bar(x - 2*width, metrics['no_isolation'], width, label='No Isolation')
#         ax.bar(x - width, metrics['label_first'], width, label='Label Words (First)')
#         ax.bar(x, metrics['label_last'], width, label='Label Words (Last)')
#         ax.bar(x + width, metrics['random_first'], width, label='Random (First)')
#         ax.bar(x + 2*width, metrics['random_last'], width, label='Random (Last)')
        
#         ax.set_ylabel('Loyalty')
#         ax.set_title(title)
#         ax.set_xticks(x)
#         ax.set_xticklabels(categories)
#         ax.legend()
        
#         plt.tight_layout()
#         if save_path:
#             plt.savefig(save_path)
#         plt.show()

#     @staticmethod
#     def plot_confusion_matrix(matrix: np.ndarray, labels: list, 
#                             title: str = "", save_path: str = None):
#         """
#         Plots confusion matrix heatmap (reproduces confusion matrix visualizations).
        
#         Args:
#             matrix: Confusion matrix values
#             labels: Category labels
#             title: Plot title
#             save_path: Path to save figure
#         """
#         plt.figure(figsize=(10, 8))
#         plt.imshow(matrix, cmap='YlOrRd')
        
#         # Add text annotations
#         for i in range(len(labels)):
#             for j in range(len(labels)):
#                 plt.text(j, i, f'{matrix[i, j]:.2f}',
#                         ha='center', va='center')
        
#         plt.colorbar()
#         plt.xticks(range(len(labels)), labels, rotation=45)
#         plt.yticks(range(len(labels)), labels)
#         plt.title(title)
        
#         plt.tight_layout()
#         if save_path:
#             plt.savefig(save_path)
#         plt.show()       
        

# class ICLResultsGenerator:
#     """Orchestrates ICL analysis and generates results using component classes."""
    
#     def __init__(self, model, tokenizer, predictor):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.predictor = predictor
        
#         # Initialize analysis components
#         self.saliency_analyzer = SaliencyAnalyzer(model)
#         self.info_blocker = InformationBlocker()
#         self.attention_modeler = AttentionModeler()
#         self.context_compressor = ContextCompressor(model)
#         self.error_analyzer = ErrorAnalyzer()
        
#         # Initialize visualization helper
#         self.visualizer = ICLVisualizer()

#     def generate_information_flow_analysis(self, 
#                                          dataset: Dict,
#                                          n_demos: int = 1,
#                                          n_samples: int = 100) -> Dict:
#         """
#         Generates complete information flow analysis (Section 2.1-2.3).
        
#         Args:
#             dataset: Dataset containing demos and test samples
#             n_demos: Number of demonstrations per class
#             n_samples: Number of test samples to analyze
        
#         Returns:
#             Dictionary containing analysis results
#         """
#         results = {}
        
#         # 1. Compute saliency metrics (Section 2.1)
#         print("Computing saliency metrics...")
#         demos, samples = self._prepare_data(dataset, n_demos, n_samples)
        
#         saliency_scores = self.saliency_analyzer.compute_saliency_scores(
#             attention_matrices=self._get_attention_matrices(samples),
#             loss=self._compute_loss(samples)
#         )
        
#         flow_metrics = self.saliency_analyzer.compute_flow_metrics(
#             saliency_scores=saliency_scores,
#             class_positions=self.predictor.get_label_positions(samples),
#             target_position=self.predictor.get_target_position(samples)
#         )
        
#         results['saliency'] = flow_metrics
        
#         # 2. Analyze information blocking (Section 2.2) 
#         print("Analyzing information blocking...")
#         blocking_results = {}
        
#         # Test label word blocking in shallow layers
#         blocked_output_shallow = self.info_blocker.block_label_connections(
#             attention_matrix=self._get_attention_matrices(samples)[0:5],  # First 5 layers
#             label_positions=self.predictor.get_label_positions(samples)
#         )
        
#         # Test label word blocking in deep layers
#         blocked_output_deep = self.info_blocker.block_label_connections(
#             attention_matrix=self._get_attention_matrices(samples)[-5:],  # Last 5 layers
#             label_positions=self.predictor.get_label_positions(samples)
#         )
        
#         # Compute loyalty metrics
#         original_output = self._get_model_output(samples)
#         blocking_results['shallow'] = self.info_blocker.compute_loyalty_metrics(
#             original_output, blocked_output_shallow
#         )
#         blocking_results['deep'] = self.info_blocker.compute_loyalty_metrics(
#             original_output, blocked_output_deep
#         )
        
#         results['blocking'] = blocking_results
        
#         # 3. Deep layer analysis (Section 2.3)
#         print("Analyzing deep layers...")
#         attention_weights = self._get_attention_weights(samples)
#         regression_results = self.attention_modeler.fit_logistic_regression(
#             attention_weights=attention_weights,
#             labels=samples['labels']
#         )
        
#         results['deep_analysis'] = regression_results
        
#         # 4. Error analysis
#         print("Analyzing errors...")
#         predictions = self._get_model_predictions(samples)
#         confusion_matrices = self.error_analyzer.compute_confusion_matrix(
#             key_vectors=self._get_key_vectors(samples),
#             predictions=predictions,
#             labels=samples['labels']
#         )
        
#         error_patterns = self.error_analyzer.analyze_error_patterns(confusion_matrices)
#         results['error_analysis'] = error_patterns
        
#         return results

#     def visualize_results(self, results: Dict, save_dir: Optional[str] = None):
#         """
#         Generates all visualizations from the paper.
        
#         Args:
#             results: Results dictionary from generate_information_flow_analysis
#             save_dir: Optional directory to save figures
#         """
#         # 1. Plot information flow (Figure 3)
#         self.visualizer.plot_information_flow(
#             layers=range(len(results['saliency']['S_wp'])),
#             Swp=results['saliency']['S_wp'],
#             Spq=results['saliency']['S_pq'],
#             Sww=results['saliency']['S_ww'],
#             title="Information Flow Analysis",
#             save_path=f"{save_dir}/figure3.png" if save_dir else None
#         )
        
#         # 2. Plot isolation impact (Figure 4)
#         isolation_metrics = {
#             'no_isolation': results['blocking']['baseline'],
#             'label_first': results['blocking']['shallow'],
#             'label_last': results['blocking']['deep'],
#             'random_first': results['blocking']['random_shallow'],
#             'random_last': results['blocking']['random_deep']
#         }
        
#         self.visualizer.plot_isolation_impact(
#             metrics=isolation_metrics,
#             title="Impact of Information Flow Isolation",
#             save_path=f"{save_dir}/figure4.png" if save_dir else None
#         )
        
#         # 3. Plot confusion matrices (Figure 6)
#         self.visualizer.plot_confusion_matrix(
#             matrix=results['error_analysis']['predicted_confusion'],
#             labels=self.predictor.get_label_names(),
#             title="Predicted Confusion Matrix",
#             save_path=f"{save_dir}/figure6a.png" if save_dir else None
#         )
        
#         self.visualizer.plot_confusion_matrix(
#             matrix=results['error_analysis']['actual_confusion'],
#             labels=self.predictor.get_label_names(),
#             title="Actual Confusion Matrix",
#             save_path=f"{save_dir}/figure6b.png" if save_dir else None
#         )

#     def _prepare_data(self, dataset: Dict, n_demos: int, n_samples: int) -> Tuple[Dict, Dict]:
#         """Prepares demonstration and test data."""
#         demos = self._sample_demonstrations(dataset, n_demos)
#         samples = self._sample_test_data(dataset, n_samples)
#         return demos, samples

#     def _get_attention_matrices(self, samples: Dict) -> torch.Tensor:
#         """Gets attention matrices from model forward pass."""
#         outputs = self.model(**samples, output_attentions=True)
#         return outputs.attentions

#     def _compute_loss(self, samples: Dict) -> torch.Tensor:
#         """Computes model loss."""
#         outputs = self.model(**samples)
#         return outputs.loss

#     def _get_model_output(self, samples: Dict) -> torch.Tensor:
#         """Gets model output logits."""
#         outputs = self.model(**samples)
#         return outputs.logits

#     def _get_model_predictions(self, samples: Dict) -> torch.Tensor:
#         """Gets model predictions."""
#         outputs = self.model(**samples)
#         return outputs.logits.argmax(dim=-1)

#     def _get_key_vectors(self, samples: Dict) -> torch.Tensor:
#         """Gets key vectors for label positions."""
#         # Implementation depends on model architecture
#         pass

#     def _get_attention_weights(self, samples: Dict) -> torch.Tensor:
#         """Gets attention weights for target position."""
#         # Implementation depends on model architecture
#         pass

#     @staticmethod
#     def _sample_demonstrations(dataset: Dict, n_demos: int) -> Dict:
#         """Samples demonstrations from dataset."""
#         # Implementation for sampling demonstrations
#         pass

#     @staticmethod
#     def _sample_test_data(dataset: Dict, n_samples: int) -> Dict:
#         """Samples test data from dataset."""
#         # Implementation for sampling test data
#         pass