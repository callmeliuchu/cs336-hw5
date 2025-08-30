
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
    return 1 





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

"""Problem (masked_normalize): Masked normalize (1 point)
Deliverable: Implement a method masked_normalize that sums over tensor elements and
normalizes by a constant while respecting a boolean mask.
The following interface is recommended:
def masked_normalize(
tensor: torch.Tensor,
mask: torch.Tensor,
normalize_constant: float,
dim: int | None = None,
) -> torch.Tensor:
11
Sum over a dimension and normalize by a constant, considering only those elements where mask
== 1.
Args:
tensor: torch.Tensor The tensor to sum and normalize.
mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.
normalize_constant: float the constant to divide by for normalization.
dim: int | None the dimension to sum along before normalization. If None, sum over all
dimensions.
Returns:
torch.Tensor the normalized sum, where masked elements (mask == 0) don’t contribute to
the sum.
To test your code, implement [adapters.run_masked_normalize]. Then run uv run pytest -k test_masked_normalize and ensure it passes."""

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    tensor_sum = torch.sum(tensor * mask, dim=dim)
    return tensor_sum / normalize_constant


"""Problem (sft_microbatch_train_step): Microbatch train step (3 points)
Deliverable: Implement a single micro-batch update for SFT, including cross-entropy loss, summing
with a mask, and gradient scaling.
The following interface is recommended:
def sft_microbatch_train_step(
policy_log_probs: torch.Tensor,
response_mask: torch.Tensor,
gradient_accumulation_steps: int,
normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
Execute a forward-and-backward pass on a microbatch.
Args:
policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
SFT policy being trained.
response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
prompt/padding.
gradient_accumulation_steps Number of microbatches per optimizer step.
normalize_constant The constant by which to divide the sum. It is fine to leave this as 1.0.
Returns:
tuple[torch.Tensor, dict[str, torch.Tensor]].
12
loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
this so we can log it.
metadata Dict with metadata from the underlying loss call, and any other statistics you
might want to log.
Implementation tips:
• You should call loss.backward() in this function. Make sure to adjust for gradient
accumulation.
To test your code, implement [adapters.run_sft_microbatch_train_step]. Then run uv
run pytest -k test_sft_microbatch_train_step and confirm it passes."""


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # policy_log_probs,response_mask,gradient_accumulation_steps,normalize_constant
    loss  = masked_normalize(-policy_log_probs,response_mask, normalize_constant, -1).mean() / gradient_accumulation_steps 
    loss.backward()
    return loss, {}


"""Problem (sft_experiment): Run SFT on the MATH dataset (2 points) (2 H100 hrs)
1. Run SFT on the reasoning SFT examples (provided in /data/a5-alignment/MATH/sft.jsonl)
using the Qwen 2.5 Math 1.5B base model, varying the number of unique examples for SFT in
14
the range {128, 256, 512, 1024}, along with using the full dataset. Tune the learning rate and
batch size to achieve at least 15% validation accuracy when using the full dataset.
Deliverable: Validation accuracy curves associated with different dataset sizes.
2. Filter the reasoning SFT examples to only include examples that produce the correct answer. Run
SFT on the (full) filtered dataset and report the size of the filtered dataset and the validation
accuracy you achieve.
Deliverable: Report the size of the dataset and the validation accuracy curve you achieve.
Compare your findings to the previous SFT experiment."""



def load_data(path):
    data_arr = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            data_arr.append(data) 
    return data_arr

def load_train_data():
    return load_data('/Users/liuchu/assignment5-alignment/data/gsm8k/train.jsonl')

def load_test_data():
    return load_data('/Users/liuchu/assignment5-alignment/data/gsm8k/test.jsonl')


import random

def sample_data(data_arr,num_samples=None):
    if num_samples is None:
        arr = data_arr
    else:
        arr =  random.sample(data_arr,num_samples)
    prompt_strs = [data['question'] for data in arr]
    output_strs = [data['answer'] for data in arr]
    return prompt_strs, output_strs

def sft_experiment():
    train_data_arr = load_train_data()
    test_data_arr = load_test_data()

    test_prompt_strs, test_output_strs = sample_data(test_data_arr)

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-1.5B')
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
    )
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

    def get_log_data(prompt,output_strs):
        res = tokenize_prompt_and_output(prompt,output_strs,tokenizer)
        input_ids = res['input_ids']
        labels = res['labels']
        logits = model(input_ids).logits
        response = model.generate(input_ids,max_new_tokens=100,do_sample=True,temperature=0.7)
        reward = reward_fn(prompt,response)
        token_entropy = compute_entropy(logits)
        response_length = len(response)
        correct_response_length = len(response)
        incorrect_response_length = len(response)
        print('prompt',prompt_strs)
        print('response',response)
        print('ground_truth',output_strs)
        print('reward',reward)
        print('token_entropy',token_entropy)
        print('response_length',response_length)
        print('correct_response_length',correct_response_length)
        print('incorrect_response_length',incorrect_response_length)

    for epoch in range(1000):
        prompt_strs, output_strs = sample_data(train_data_arr,8)
        res = tokenize_prompt_and_output(prompt_strs,output_strs,tokenizer)
        policy_log_probs = get_response_log_probs(model,res['input_ids'],res['labels'],return_token_entropy=True)['log_probs']
        response_mask = res['response_mask']
        optimizer.zero_grad()
        loss, metadata = sft_microbatch_train_step(policy_log_probs,response_mask,gradient_accumulation_steps=1,normalize_constant=1.0)
        optimizer.step()
        print('loss',loss)
        """Logging generations in-the-loop. It’s always good practice to do some in-the-loop logging that involves
        generation from your model, and reasoning SFT/RL is no exception. Write a function log_generations
        that will prompt your model to generate responses for some given prompts (e.g., sampled from the validation
        set). It’s a good idea to log at least the following for each example:
        1. The input prompt.
        2. The response generated by the SFT/RL model.
        3. The ground-truth answer.
        4. The reward information, including format, answer, and total reward.
        5. The average token entropy of the response.
        6. The average response length, average response length for correct responses, and average response length
        for incorrect responses."""
        #### 
        # print('logging..........')
        # get_log_data(prompt_strs,output_strs)
        # print('logging done..........')



# sft_experiment()


"""Problem (compute_group_normalized_rewards): Group normalization (2 points)
Deliverable: Implement a method compute_group_normalized_rewards that calculates raw
rewards for each rollout response, normalizes them within their groups, and returns both the
normalized and raw rewards along with any metadata you think is useful.
The following interface is recommended:
def compute_group_normalized_rewards(
reward_fn,
rollout_responses,
repeated_ground_truths,
group_size,
advantage_eps,
normalize_by_std,
):
Compute rewards for each group of rollout responses, normalized by the group size.
Args:
reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against
the ground truths, producing a dict with keys "reward", "format_reward", and
"answer_reward".
rollout_responses: list[str] Rollouts from the policy. The length of this list is
rollout_batch_size = n_prompts_per_rollout_batch * group_size.
repeated_ground_truths: list[str] The ground truths for the examples. The length of this
list is rollout_batch_size, because the ground truth for each example is repeated
group_size times.
group_size: int Number of responses per question (group).
advantage_eps: float Small constant to avoid division by zero in normalization.
normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise
subtract only the group mean.
Returns:
tuple[torch.Tensor, torch.Tensor, dict[str, float]].
advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout
response.
22
raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout
response.
metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards).
To test your code, implement [adapters.run_compute_group_normalized_rewards]. Then,
run the test with uv run pytest -k test_compute_group_normalized_rewards and make
sure your implementation passes it."""


def compute_group_normalized_rewards(reward_fn,rollout_responses,repeated_ground_truths,group_size,advantage_eps,normalize_by_std):
    # producing a dict with keys "reward", "format_reward", and "answer_reward".
    rewards_arr = []
    format_rewards_arr = []
    answer_rewards_arr = []
    for resp,truth in zip(rollout_responses,repeated_ground_truths):
        rewards_dict = reward_fn(resp,truth)
        rewards_arr.append(rewards_dict['reward'])
        format_rewards_arr.append(rewards_dict['format_reward'])
        answer_rewards_arr.append(rewards_dict['answer_reward'])
    rewards = torch.tensor(rewards_arr)
    format_rewards = torch.tensor(format_rewards_arr)
    answer_rewards = torch.tensor(answer_rewards_arr)
    rewards = rewards.reshape(-1,group_size) # (rollout_batch_size, group_size)
    rewards_mean = rewards.mean(dim=-1,keepdim=True) # (rollout_batch_size,1)
    rewards_std = rewards.std(dim=-1,keepdim=True) # (rollout_batch_size,1)    
    if normalize_by_std:
        rewards_normalized = (rewards - rewards_mean) / (rewards_std + advantage_eps) # (rollout_batch_size,)
    else:
        rewards_normalized = (rewards - rewards_mean) # (rollout_batch_size,)
    rewards_normalized = rewards_normalized.reshape(-1) # (rollout_batch_size,)
    rewards = rewards.reshape(-1) # (rollout_batch_size,)
    format_rewards = format_rewards.reshape(-1) # (rollout_batch_size,)
    answer_rewards = answer_rewards.reshape(-1) # (rollout_batch_size,)
    return rewards_normalized, rewards, {
        'rewards_mean': rewards_mean,
        'rewards_std': rewards_std,
        'format_rewards': format_rewards,
        'answer_rewards': answer_rewards,
    }



"""Problem (compute_naive_policy_gradient_loss): Naive policy gradient (1 point)
Deliverable: Implement a method compute_naive_policy_gradient_loss that computes the
per-token policy-gradient loss using raw rewards or pre-computed advantages.
The following interface is recommended:
def compute_naive_policy_gradient_loss(
raw_rewards_or_advantages: torch.Tensor,
policy_log_probs: torch.Tensor,
) -> torch.Tensor:
Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either
the raw reward or an already-normalized advantage.
Args:
raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar
reward/advantage for each rollout response.
policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for
each token.
Returns:
torch.Tensor Shape (batch_size, sequence_length), the per-token policy-gradient loss (to
be aggregated across the batch and sequence dimensions in the training loop).
Implementation tips:
• Broadcast the raw_rewards_or_advantages over the sequence_length dimension.
To test your code, implement [adapters.run_compute_naive_policy_gradient_loss]. Then
run uv run pytest -k test_compute_naive_policy_gradient_loss and ensure the test
passes."""

def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
    ) -> torch.Tensor:
    return - raw_rewards_or_advantages * policy_log_probs

"""Problem (compute_grpo_clip_loss): GRPO-Clip loss (2 points)
Deliverable: Implement a method compute_grpo_clip_loss that computes the per-token
GRPO-Clip loss.
The following interface is recommended:
def compute_grpo_clip_loss(
advantages: torch.Tensor,
policy_log_probs: torch.Tensor,
old_log_probs: torch.Tensor,
cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
Args:
advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log
probs from the policy being trained.
old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs
from the old policy.
cliprange: float Clip parameter ϵ (e.g. 0.2).
Returns:
tuple[torch.Tensor, dict[str, torch.Tensor]].
loss torch.Tensor of shape (batch_size, sequence_length), the per-token clipped
loss.
metadata dict containing whatever you want to log. We suggest logging whether each
token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of
the min was lower than the LHS.
Implementation tips:
• Broadcast advantages over sequence_length.
To test your code, implement [adapters.run_compute_grpo_clip_loss]. Then run uv run
pytest -k test_compute_grpo_clip_loss and ensure the test passes."""


def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratio = torch.exp(policy_log_probs - old_log_probs)
    return -torch.min(ratio * advantages,torch.clip(ratio,1-cliprange,1+cliprange) * advantages) , {}

    
"""Problem (compute_policy_gradient_loss): Policy-gradient wrapper (1 point)
Deliverable: Implement compute_policy_gradient_loss, a convenience wrapper that dispatches
to the correct loss routine (no_baseline, reinforce_with_baseline, or grpo_clip) and returns
both the per-token loss and any auxiliary statistics.
The following interface is recommended:
def compute_policy_gradient_loss(
policy_log_probs: torch.Tensor,
loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
raw_rewards: torch.Tensor | None = None,
advantages: torch.Tensor | None = None,
old_log_probs: torch.Tensor | None = None,
cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
Select and compute the desired policy-gradient loss.
Args:
policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
policy being trained.
loss_type One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
raw_rewards Required if loss_type == "no_baseline"; shape (batch_size, 1).
advantages Required for "reinforce_with_baseline" and "grpo_clip"; shape
(batch_size, 1).
old_log_probs Required for "grpo_clip"; shape (batch_size, sequence_length).
cliprange Required for "grpo_clip"; scalar ϵ used for clipping.
Returns:
tuple[torch.Tensor, dict[str, torch.Tensor]].
loss (batch_size, sequence_length), per-token loss.
metadata dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
Implementation tips:
• Delegate to compute_naive_policy_gradient_loss or compute_grpo_clip_loss.
• Perform argument checks (see assertion pattern above).
• Aggregate any returned metadata into a single dict.
To test your code, implement [adapters.run_compute_policy_gradient_loss]. Then run uv
run pytest -k test_compute_policy_gradient_loss and verify it passes."""

from typing import Literal


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == 'no_baseline':
        return compute_naive_policy_gradient_loss(raw_rewards,policy_log_probs), {}
    elif loss_type == 'reinforce_with_baseline':
        return compute_naive_policy_gradient_loss(advantages,policy_log_probs), {}
    else:
        loss, _= compute_grpo_clip_loss(advantages,policy_log_probs,old_log_probs,cliprange)
        return loss, {}