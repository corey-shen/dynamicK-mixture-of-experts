# Utilizing Llama-3.2-1B for local testing
# https://huggingface.co/meta-llama/Llama-3.2-1B
# NOTE: this is a gated model, login with a HF token with gated access permission

from transformers import pipeline
import torch
from torchvision import models
from torchsummary import summary

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
model_name = model.config.name_or_path

if torch.cuda.is_available():   # if we want to save device, set device = torch.device(param)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
vgg = models.vgg16().to(device)
summary(vgg, (3, 224, 224))  # Outputs model's layers, output shapes, and number of parameters

def generate_response(prompt, max_length=200):
    result = pipe(prompt, max_length=max_length, do_sample=True, temperature=0.7, truncation=True)   # do_sample = True selects token based on probability dist., not greedy approach
    return result[0]['generated_text']

test_prompt = "Explain the difference between machine learning and deep learning"
response = generate_response(test_prompt)


# === Print Results ===
print(f"\nModel Name: {model_name}\n")
print(tokens_per_second)
print(f"OUTPUT: \n{response}")