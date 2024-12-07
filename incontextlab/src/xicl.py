# my_sdk/sdk.py
from typing import List, Dict
from .models.amplify_model import AMPLIFYWrapper
from .models.latent_concept_model import LatentConceptWrapper
from .models.information_flow_model import InformationFlowWrapper
from .models.semantic_head_model import SemanticHeadWrapper

class MySDK:
    def __init__(self):
        self.amplify = AMPLIFYWrapper()
        self.latent_concept = LatentConceptWrapper()
        self.information_flow = InformationFlowWrapper()
        self.semantic_head = SemanticHeadWrapper()

    def run(self, data: List[Dict]) -> Dict:
        if not data:
            raise ValueError("No data provided.")

        # All data assumed to have same task_type for simplicity
        task_type = data[0]["metadata"]["task_type"]
        task_name = data[0]["metadata"].get("task_name", None)

        # Route to models depending on task_type & task_name
        if task_type == "classification":
            # For demonstration: if sentiment/emotion/sarcasm -> amplify
            if task_name in ["sentiment", "emotion", "sarcasm"]:
                return self.amplify.run(data)
            else:
                # Otherwise use latent_concept as an example
                return self.latent_concept.run(data)

        elif task_type == "relation_extraction":
            # Use semantic_head to analyze relations
            return self.semantic_head.run(data)

        elif task_type == "information_flow":
            # If defined as a special type for analyzing info flow
            return self.information_flow.run(data)

        else:
            raise ValueError(f"Unknown task_type: {task_type}")
