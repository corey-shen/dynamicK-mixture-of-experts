# Utilizing Llama-3.2-1B for local testing
# https://huggingface.co/meta-llama/Llama-3.2-1B
# NOTE: this is a gated model, login with a HF token with gated access permission

from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

def generate_response(prompt, max_length=200):
    result = pipe(prompt, max_length=max_length, do_sample=True, temperature=0.7, truncation=True)   # do_sample = True selects token based on probability dist., not greedy approach
    return result[0]['generated_text']

test_prompt = "Explain the difference between machine learning and deep learning"
response = generate_response(test_prompt)
print(response)