
"""Problem (math_baseline): 4 points
(a) Write a script to evaluate Qwen 2.5 Math 1.5B zero-shot performance on MATH. This script
should (1) load the MATH validation examples from /data/a5-alignment/MATH/validation.jsonl,
(2) format them as string prompts to the language model using the r1_zero prompt, and (3) gen-
erate outputs for each example. This script should also (4) calculate evaluation metrics and
(5) serialize the examples, model generations, and corresponding evaluation scores to disk for
analysis in subsequent problems.
It might be helpful for your implementation to include a method evaluate_vllm with arguments
similar to the following, as you will be able to reuse it later:
def evaluate_vllm(
vllm_model: LLM,
reward_fn: Callable[[str, str], dict[str, float]],
prompts: List[str],
eval_sampling_params: SamplingParams
) -> None:
Evaluate a language model on a list of prompts,
compute evaluation metrics, and serialize results to disk.
Deliverable: A script to evaluate baseline zero-shot MATH performance.
(b) Run your evaluation script on Qwen 2.5 Math 1.5B. How many model generations fall into each
of the following categories: (1) correct with both format and answer reward 1, (2) format reward
1 and answer reward 0, (3) format reward 0 and answer reward 0? Observing at least 10 cases
where format reward is 0, do you think the issue is with the base model’s output, or the parser?
Why? What about in (at least 10) cases where format reward is 1 but answer reward is 0?
Deliverable: Commentary on the model and reward function performance, including examples
of each category.
(c) How well does the Qwen 2.5 Math 1.5B zero-shot baseline perform on MATH?
Deliverable: 1-2 sentences with evaluation metrics."""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os




def test():
    # Load model and tokenizer with CPU optimization
    print("Loading model and tokenizer with accelerate (CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
        torch_dtype=torch.float32,  # Use float32 for CPU compatibility
        device_map=None  # Don't use device_map on CPU
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")

    def test_model(prompt, max_new_tokens=100):
        """Test the model with a given prompt"""
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Test the model with a simple example
    print("Testing with simple math problem:")
    test_prompt = "What is 5 + 3? Please show your work."
    response = test_model(test_prompt)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response}\n")


from typing import Callable, List


def evaluate_vllm(
    model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    tokenizer = AutoTokenizer.from_pretrained(model)
    for prompt in prompts:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt")
            response = model.generate(**inputs, **eval_sampling_params)
            reward = reward_fn(prompt, tokenizer.decode(response[0], skip_special_tokens=True))
            print(reward)


def reward_fn(prompt, response):
    return 1 if response.strip() == prompt.strip() else 0





"""Problem (tokenize_prompt_and_output): Prompt and output tokenization (2 points)
Deliverable: Implement a method tokenize_prompt_and_output that tokenizes the question and
output separately, concatenates them together, and constructs a response_mask. The following
interface is recommended:
def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer): Tokenize the
prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for
other tokens (prompt or padding).
Args:
prompt_strs: list[str] List of prompt strings.
output_strs: list[str] List of output strings.
tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.
Returns:
dict[str, torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of
the tokenized prompt and output strings. Then the returned dictionary should have the
following keys:
input_ids torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
the tokenized prompt and output strings, with the final token sliced off.
labels torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
shifted input ids, i.e., the input ids without the first token.
response_mask torch.Tensor of shape (batch_size, max(prompt_and_output_lens) -
1): a mask on the response tokens in the labels.
To test your code, implement [adapters.run_tokenize_prompt_and_output]. Then, run the
test with uv run pytest -k test_tokenize_prompt_and_output and make sure your
implementation passes it."""

import torch
from typing import List


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    prompt_tokens = tokenizer(prompt_strs)['input_ids']
    output_tokens = tokenizer(output_strs)['input_ids']

    batch_sz = len(prompt_tokens)

    prompt_and_output_lens = [len(p) + len(o) for p, o in zip(prompt_tokens, output_tokens)]
    padded_len = max(prompt_and_output_lens)

    input_ids = torch.empty((batch_sz, padded_len - 1), dtype=torch.long)
    labels = torch.empty((batch_sz, padded_len - 1), dtype=torch.long)
    response_mask = torch.zeros((batch_sz, padded_len - 1), dtype=torch.bool)

    for i, (p_toks, o_toks) in enumerate(zip(prompt_tokens, output_tokens)):
        p_o_concat = torch.tensor(p_toks + o_toks)
        concat_len = len(p_o_concat)
        p_o_concat_padded = F.pad(p_o_concat, (0, padded_len - concat_len), 'constant', tokenizer.eos_token_id)

        input_ids[i] = p_o_concat_padded[:-1]
        labels[i] = p_o_concat_padded[1:]

        o_start = len(p_toks) - 1
        o_end = concat_len - 1
        response_mask[i, o_start:o_end] = True
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'response_mask': response_mask,
    }





"""Problem (compute_entropy): Per-token entropy (1 point)
Deliverable: Implement a method compute_entropy that computes the per-token entropy of
next-token predictions.
The following interface is recommended:
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
Args:
logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
containing unnormalized logits.
Returns:
torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
prediction.
Note: you should use a numerically stable method (e.g., using logsumexp) to avoid overflow.
To test your code, implement [adapters.run_compute_entropy]. Then run uv run pytest -k test_compute_entropy and ensure your implementation passes."""

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    next_token_logits = logits
    probs = F.softmax(next_token_logits,dim=-1)
    total = torch.sum(next_token_logits * probs,dim=-1)
    return torch.logsumexp(next_token_logits,dim=-1) - total


"""Problem (get_response_log_probs): Response log-probs (and entropy) (2 points)
Deliverable: Implement a method get_response_log_probs that gets per-token conditional
log-probabilities (given the previous tokens) from a causal language model, and optionally the
entropy of the model’s next-token distribution.
The following interface is recommended:
10
def get_response_log_probs(
model: PreTrainedModel,
input_ids: torch.Tensor,
labels: torch.Tensor,
return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
Args:
model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device
and in inference mode if gradients should not be computed).
input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt +
response tokens as produced by your tokenization method.
labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your
tokenization method.
return_token_entropy: bool If True, also return per-token entropy by calling
compute_entropy.
Returns:
dict[str, torch.Tensor].
"log_probs" shape (batch_size, sequence_length), conditional log-probabilities
log pθ (xt | x<t).
"token_entropy" optional, shape (batch_size, sequence_length), per-token entropy
for each position (present only if return_token_entropy=True).
Implementation tips:
• Obtain logits with model(input_ids).logits.
To test your code, implement [adapters.run_get_response_log_probs]. Then run uv run
pytest -k test_get_response_log_probs and ensure the test passes."""


def get_response_log_probs(model,input_ids,labels,return_token_entropy=False):
    ### input_ids (batch_size, sequence_length)
    ### labels (batch_size, sequence_length)
    logits = model(input_ids).logits # (batch_size, sequence_length, vocab_size)
    logprobs = torch.log_softmax(logits,dim=-1)
    labels_logprobs = logprobs.gather(dim=-1,index=labels.unsqueeze(-1)).squeeze(-1) # (batch_size, sequence_length)
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
    return {
        "log_probs": labels_logprobs,
        "token_entropy": token_entropy if return_token_entropy else None
    }