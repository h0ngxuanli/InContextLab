from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import matplotlib.pyplot as plt

# Proxy Model Class
class ProxyModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.explainer = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

    def explain(self, text: str) -> Dict[str, float]:
        """
        Generate attribution-like scores for each label.
        """
        predictions = self.explainer(text, return_all_scores=True)[0]
        return {item['label']: item['score'] for item in predictions}


# Misclassification Confidence Score Calculator
class MisclassificationScore:
    @staticmethod
    def calculate(predicted: Dict[str, float], true_label: str) -> float:
        """
        Calculate MCS: the difference between max incorrect and true label scores.
        """
        true_score = predicted.get(true_label, 0)
        max_false_score = max([score for label, score in predicted.items() if label != true_label], default=0)
        return max_false_score - true_score


# Few-Shot Prompt Class
class FewShotPrompt:
    def __init__(self, proxy_model: ProxyModel, top_k_keywords: int):
        self.proxy_model = proxy_model
        self.top_k_keywords = top_k_keywords

    def generate_rationale(self, text: str, label: str) -> str:
        """
        Generate a rationale using top-k keywords based on attribution scores.
        """
        predictions = self.proxy_model.explain(text)
        sorted_keywords = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:self.top_k_keywords]
        keywords = [word for word, _ in sorted_keywords]
        return f"The key words: {', '.join(keywords)} are crucial clues for predicting {label} as the correct answer."

    def construct_prompt(self, samples: List[Tuple[str, str]]) -> str:
        """
        Construct the full few-shot prompt.
        """
        prompt = ""
        for text, label in samples:
            rationale = self.generate_rationale(text, label)
            prompt += f"Input: {text}\nRationale: {rationale}\nLabel: {label}\n\n"
        return prompt


# Amplify Framework Class
class AmplifyFramework:
    def __init__(self, proxy_model_name: str, top_k_samples: int, top_k_keywords: int):
        self.proxy_model = ProxyModel(proxy_model_name)
        self.top_k_samples = top_k_samples
        self.top_k_keywords = top_k_keywords

    def select_samples(self, validation_set: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Select top-k samples with the highest MCS.
        """
        scored_samples = []
        for text, label in validation_set:
            predictions = self.proxy_model.explain(text)
            mcs = MisclassificationScore.calculate(predictions, label)
            scored_samples.append((text, label, mcs))
        
        scored_samples.sort(key=lambda x: x[2], reverse=True)
        return [(sample[0], sample[1]) for sample in scored_samples[:self.top_k_samples]]

    def run(self, validation_set: List[Tuple[str, str]]) -> str:
        """
        Execute the AMPLIFY pipeline and construct a prompt.
        """
        selected_samples = self.select_samples(validation_set)
        prompt_generator = FewShotPrompt(self.proxy_model, self.top_k_keywords)
        return prompt_generator.construct_prompt(selected_samples)


# Visualization Class
class ExperimentVisualizer:
    @staticmethod
    def plot_mcs_distribution(mcs_scores: List[float], title: str = "MCS Distribution"):
        """
        Visualize the MCS distribution of selected samples.
        """
        plt.hist(mcs_scores, bins=10, color='blue', alpha=0.7)
        plt.title(title)
        plt.xlabel("MCS")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()


# Main Example
if __name__ == "__main__":
    # Example validation data
    validation_data = [
        ("The movie was fantastic and had great visuals.", "Positive"),
        ("The food was terrible and the service was slow.", "Negative"),
        ("The book was engaging but the ending was disappointing.", "Neutral"),
        ("The product quality was outstanding and delivery was quick.", "Positive"),
        ("The app crashed frequently and was full of bugs.", "Negative"),
    ]

    # Initialize AMPLIFY framework
    amplify = AmplifyFramework(proxy_model_name="distilbert-base-uncased", top_k_samples=3, top_k_keywords=3)

    # Run framework
    few_shot_prompt = amplify.run(validation_data)

    # Display the prompt
    print("Constructed Few-Shot Prompt:\n")
    print(few_shot_prompt)

    # Visualize MCS distribution
    proxy_model = amplify.proxy_model
    mcs_scores = [
        MisclassificationScore.calculate(proxy_model.explain(text), label)
        for text, label in validation_data
    ]
    ExperimentVisualizer.plot_mcs_distribution(mcs_scores)
