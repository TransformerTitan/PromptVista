import pytest
from promptvista.runner import PromptRunner
from promptvista.models import ModelWrapper

class DummyModel:
    def generate(self, prompt, max_tokens=50):
        return f"Dummy response for: {prompt}"

def test_prompt_runner():
    dummy_models = {"dummy": DummyModel()}
    runner = PromptRunner(dummy_models)
    prompt = "Hello world"
    outputs = runner.run_prompt(prompt)
    assert "dummy" in outputs
    assert outputs["dummy"] == "Dummy response for: Hello world"
