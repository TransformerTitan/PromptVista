from .models import ModelWrapper

class PromptRunner:
    def __init__(self, models):
        """
        models: dict of {model_name: ModelWrapper instance}
        """
        self.models = models

    def run_prompt(self, prompt, max_tokens=50):
        results = {}
        for name, model in self.models.items():
            output = model.generate(prompt, max_tokens=max_tokens)
            results[name] = output
        return results
