from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import openai

class ModelWrapper:
    def __init__(self, model_name, device='cpu'):
        self.device = device
        self.model_name = model_name
        self.is_openai = model_name.startswith("openai:")

        if self.is_openai:
            openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace or set env var before running
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            self.model.eval()

    def generate(self, prompt, max_tokens=50):
        if self.is_openai:
            model_id = self.model_name.split("openai:")[1]
            response = openai.Completion.create(
                engine=model_id,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                n=1,
                stop=None
            )
            return response.choices[0].text.strip()
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7
                )
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt prefix from the generated text
            return decoded[len(prompt):].strip()
