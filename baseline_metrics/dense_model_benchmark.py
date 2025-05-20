# Utilizing Llama-3.2-1B for local testing
# https://huggingface.co/meta-llama/Llama-3.2-1B
# NOTE: this is a gated model, login with a HF token with gated access permission

'''
TODO: test cuda timing
- Possibly split script into a big if-else statement for CUDA/CPU computation
'''

from transformers import pipeline
import torch, time
from torchvision import models
from torchsummary import summary
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
model_name = model.config.name_or_path

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    model = model.to(device)
    print("Using MPS acceleration")
else:
    device = torch.device('cpu')
compute_time = 0.0

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B", device = device)

vgg_cpu = models.vgg16()
summary(vgg_cpu, (3, 224, 224), device = "cpu")  # Outputs model's layers, output shapes, and number of parameters
vgg = vgg_cpu.to(device)    # Set back to device

def generate_response(prompt, max_new_tokens=500): 
    global compute_time
    if device.type == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        result = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, truncation=True)   # do_sample = True selects token based on probability dist., not greedy approach
        end.record()

        torch.cuda.synchronize()
        compute_time = start.elapsed_time(end) / 1000
    else:
        start = time.perf_counter()
        result = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, truncation=True)
        end = time.perf_counter()
        compute_time = (end - start) 
    return result[0]['generated_text']

def main():
    small_prompt = "blah"
    _ = pipe(small_prompt, max_new_tokens=1, do_sample=False)
    short_execution = generate_response(small_prompt, 1)
    print(f"Short execution output to absorb python overhead: {short_execution}")

    test_prompt = "What is the difference between reinforcement learning and machine learning?"
    response = generate_response(test_prompt)
    tokens = tokenizer.encode(response)
    token_count = len(tokens)
    print(f"Token count: {token_count}")

    # === Print Results ===
    print(f"\nModel Name: {model_name}")
    print(f"Compute time: {compute_time:.4f} seconds\n")
    print(f"OUTPUT: \n{response}")

if __name__ == "__main__":
    main()