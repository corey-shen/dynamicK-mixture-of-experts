# Utilizing Llama-3.2-1B for local testing
# https://huggingface.co/meta-llama/Llama-3.2-1B
# NOTE: this is a gated model, login with a HF token with gated access permission

from transformers import pipeline
import torch, time
from torchvision import models
from torchsummary import summary
from transformers import AutoTokenizer, AutoModelForCausalLM

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
model_name = model.config.name_or_path

if torch.cuda.is_available():   # if we want to save device, set device = torch.device(param)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

compute_time = 0.0

vgg = models.vgg16().to(device)
summary(vgg, (3, 224, 224))  # Outputs model's layers, output shapes, and number of parameters

def generate_response(prompt, max_length=200):
    global compute_time
    if device.type == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        result = pipe(prompt, max_length=max_length, do_sample=True, temperature=0.7, truncation=True)   # do_sample = True selects token based on probability dist., not greedy approach
        end.record()

        torch.cuda.synchronize()
        compute_time = start.elapsed_time(end) / 100
    else:
        start = time.perf_counter()
        result = pipe(prompt, max_length=max_length, do_sample=True, temperature=0.7, truncation=True)
        end = time.perf_counter()
        compute_time = (end - start) # Records in microseconds
    return result[0]['generated_text']
    
test_prompt = "Explain the difference between machine learning and deep learning"
response = generate_response(test_prompt)

# === Print Results ===
print(f"\nModel Name: {model_name}")
print(f"Compute time: {compute_time:.4f} seconds\n")    # Seems to only measure CPU time, excluding tokenization process
print(f"OUTPUT: \n{response}")