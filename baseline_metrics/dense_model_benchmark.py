
'''
NOTE: this is a gated model, login with a HF token with gated access permission
Utilizing Llama-3.2-1B for local testing
https://huggingface.co/meta-llama/Llama-3.2-1B


TODO: The following are a list of immediate todos, the doc should contain 
ALL baseline metrics required (there is a chart for tracking implementation of metrics on page 10 of "Research Proposal")

- Implement time_cpu_inference() and time_cuda_inference() method  
- Test CUDA timing accuracy
- Consider adding warmup runs to reduce overhead
- Add error handling for model loading
- Fix attention masking bug (currently does not affect the model running)
- Implement all other baseline metrics (FLOPS/sec, throughout, capacity, etc.)
'''

from transformers import pipeline
import torch, time
from torchvision import models
from torchsummary import summary
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_device():   
    # Determine the best available device for inference
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA acceleration")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS acceleration")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
device = get_device()
model = model.to(device)    # Move model to device
model_name = model.config.name_or_path

class ModelBenchmark:
    def __init__(self, model, tokenizer, device, model_name):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name
        self.compute_time = 0.0
        self.tokens_per_second = 0.0

        ''' TODO Fix the following error:
        "The attention mask is not set and cannot be inferred from input because pad token is same as eos token. 
        As a consequence, you may observe unexpected behavior. 
        Please pass your input's `attention_mask` to obtain reliable results."
        '''
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token # Try unk_token
    
    def run_inference(self, prompt, max_new_tokens=500):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)
        input_token_count = len(input_ids[0])

        if self.device.type == 'cuda':
            self.compute_time = self.time_cuda_inference(input_ids, max_new_tokens)
        else:
            self.compute_time = self.time_cpu_inference(input_ids, max_new_tokens)

        output = self.model.generate(
                input_ids, max_new_tokens = max_new_tokens, do_sample=True, temperature = 0.7, pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        output_token_count = len(output[0])
        new_tokens = output_token_count - input_token_count
        self.tokens_per_second = new_tokens / self.compute_time if self.compute_time > 0 else 0
        return response        
    
    def tokens_per_second(self, prompt, max_new_tokens=500):
        return self.tokens_per_second
    
    def time_cuda_inference(self, input_ids, max_new_tokens):
        '''
        TODO: implement time inference using CUDA events
        Should use:
        - torch.cuda.Event(enable_timing=True) 
        - start.record() and end.record()
        - torch.cuda.synchronize()
        - start.elapsed_time(end) / 1000 to get seconds
        
        Return: compute time in seconds
        '''
        return 0

    def time_cpu_inference(self, input_ids, max_new_tokens):
        '''
        TODO: Implement time inference using time.perf_counter()

        There is code on the master branch on the repo, should be very similar, although the implementation 
        was for the pipeline API (ensure it works for .generate() if you intend to copy)
        
        Return: compute time in seconds
        '''
        return 0

    def print_metrics(self):
        print(f"\n{'='*50}")
        print(f"BENCHMARK RESULTS")
        print(f"Model Name: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Compute time: {self.compute_time:.4f} seconds")
        print(f"Tokens per second: {self.tokens_per_second:.2f}")

def main():
    '''
    # EXECUTION OVERHEAD. Previously implemented for pipeline API implementation, 
    # check if running it before running the actual test_prompt reduces overhead. 
    # If so, run this (check if it needs to be converted to using .generate()) before 
    # measuring benchmarks for the actual prompt
    # 
    # If overhead function DOES indeed reduce overhead, create a function within the 
    # ModelBenchmark class and call it before measuring any benchmarks

    small_prompt = "blah"
    _ = pipe(small_prompt, max_new_tokens=1, do_sample=False)
    short_execution = generate_response(small_prompt, 1)
    print(f"Short execution output to absorb python overhead: {short_execution}")
    '''
    benchmark = ModelBenchmark(model, tokenizer, device, model_name)
    test_prompt = "What is machine learning?"
    print("OUTPUT:")
    print(benchmark.run_inference(test_prompt))
    print()
    benchmark.print_metrics()

if __name__ == "__main__":
    main()