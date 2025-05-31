import streamlit as st
from .runner import PromptRunner
from .models import ModelWrapper

def launch_ui():
    st.title("PromptVista — Multi-Model Prompt Experimentation Toolkit")

    models_list = {
        "GPT-2": "gpt2",
        "GPT-Neo 125M": "EleutherAI/gpt-neo-125M",
        "OpenAI GPT-3": "openai:text-davinci-003"
    }

    selected_models = st.multiselect("Select models to run prompts on", list(models_list.keys()), default=["GPT-2"])

    prompt = st.text_area("Enter your prompt", "Once upon a time,")

    max_tokens = st.slider("Max tokens to generate", min_value=10, max_value=200, value=50)

    if st.button("Run Prompt"):
        if not selected_models:
            st.warning("Please select at least one model.")
            return

        with st.spinner("Generating outputs..."):
            models = {}
            for model_name in selected_models:
                models[model_name] = ModelWrapper(models_list[model_name], device="cpu")

            runner = PromptRunner(models)
            results = runner.run_prompt(prompt, max_tokens=max_tokens)

            for model_name, output in results.items():
                st.subheader(f"Output from {model_name}")
                st.write(output)
