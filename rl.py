
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
import gc
from vllm import LLM

# 设置环境变量
# 双GPU配置：GPU 0用于推理，GPU 1用于训练
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# 设置CUDA调试环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA操作，便于调试


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


def evaluate_model(
    model,
    tokenizer,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    max_new_tokens=100,
    temperature=0.7
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    model.eval()
    for prompt in prompts:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            response = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(response[0], skip_special_tokens=True)
            reward = reward_fn(prompt, generated_text)
            print(reward)








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


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer, device=None):
    prompt_tokens = tokenizer(prompt_strs)['input_ids']
    output_tokens = tokenizer(output_strs)['input_ids']

    batch_sz = len(prompt_tokens)

    prompt_and_output_lens = [len(p) + len(o) for p, o in zip(prompt_tokens, output_tokens)]
    padded_len = max(prompt_and_output_lens)

    # 如果指定了设备，在指定设备上创建tensor
    if device is not None:
        input_ids = torch.empty((batch_sz, padded_len - 1), dtype=torch.long, device=device)
        labels = torch.empty((batch_sz, padded_len - 1), dtype=torch.long, device=device)
        response_mask = torch.zeros((batch_sz, padded_len - 1), dtype=torch.bool, device=device)
    else:
        input_ids = torch.empty((batch_sz, padded_len - 1), dtype=torch.long)
        labels = torch.empty((batch_sz, padded_len - 1), dtype=torch.long)
        response_mask = torch.zeros((batch_sz, padded_len - 1), dtype=torch.bool)

    for i, (p_toks, o_toks) in enumerate(zip(prompt_tokens, output_tokens)):
        if device is not None:
            p_o_concat = torch.tensor(p_toks + o_toks, device=device)
        else:
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
    return load_data('data/gsm8k/train.jsonl')

def load_test_data():
    return load_data('data/gsm8k/test.jsonl')


import random



def format_question(question):
    prompt = f"""here is the question: {question} think step by step and answer the question. and use <answer> </answer> to wrap the answer."""
    return prompt.format(question=question)

def sample_data(data_arr,num_samples=None):
    if num_samples is None:
        arr = data_arr
    else:
        arr =  random.sample(data_arr,num_samples)
    prompt_strs = [format_question(data['question']) for data in arr]
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
        # 只打印关键数据
        print(f'Step {epoch}: loss={loss.item():.6f}, reward={reward:.3f}')

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


def compute_group_normalized_rewards(reward_fn,rollout_responses,repeated_ground_truths,group_size,advantage_eps,normalize_by_std,device=None):
    # producing a dict with keys "reward", "format_reward", and "answer_reward".
    rewards_arr = []
    format_rewards_arr = []
    answer_rewards_arr = []
    for resp,truth in zip(rollout_responses,repeated_ground_truths):
        rewards_dict = reward_fn(resp,truth)
        rewards_arr.append(rewards_dict['reward'])
        format_rewards_arr.append(rewards_dict['format_reward'])
        answer_rewards_arr.append(rewards_dict['answer_reward'])
    
    # 只打印关键统计信息
    print(f'Rewards: mean={sum(rewards_arr)/len(rewards_arr):.3f}, std={torch.tensor(rewards_arr).std().item():.3f}')
    if device is not None:
        rewards = torch.tensor(rewards_arr, device=device)
        format_rewards = torch.tensor(format_rewards_arr, device=device)
        answer_rewards = torch.tensor(answer_rewards_arr, device=device)
    else:
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
    rewards_normalized = rewards_normalized.reshape(-1,1) # (rollout_batch_size,)
    rewards = rewards.reshape(-1,1) # (rollout_batch_size,)
    format_rewards = format_rewards.reshape(-1,1) # (rollout_batch_size,)
    answer_rewards = answer_rewards.reshape(-1,1) # (rollout_batch_size,)
    return rewards_normalized, rewards, {
        'rewards_mean': rewards_mean,
        'rewards_std': rewards_std,
        'format_rewards': format_rewards,
        'answer_rewards': answer_rewards,
        'raw_rewards': rewards
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
    clipped_ratio = torch.clamp(ratio, 1-cliprange, 1+cliprange)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    return loss, {}

    
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

"""Problem (masked_mean): Masked mean (1 point)
Deliverable: Implement a method masked_mean that averages tensor elements while respecting a
boolean mask.
The following interface is recommended:
def masked_mean(
tensor: torch.Tensor,
mask: torch.Tensor,
dim: int | None = None,
) -> torch.Tensor:
Compute the mean of tensor along a given dimension, considering only those elements where
mask == 1.
Args:
tensor: torch.Tensor The data to be averaged.
mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.
dim: int | None Dimension over which to average. If None, compute the mean over all
masked elements.
Returns:
torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics.
To test your code, implement [adapters.run_masked_mean]. Then run uv run pytest -k test_masked_mean and ensure it passes."""

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    ) -> torch.Tensor:
    if dim is None:
        masked = tensor * mask
        return masked.sum() / mask.sum()
    masked = tensor * mask
    masked_sum = masked.sum(dim=dim)
    length = mask.sum(dim=dim)
    return masked_sum / length


"""Problem (grpo_microbatch_train_step): Microbatch train step (3 points)
Deliverable: Implement a single micro-batch update for GRPO, including policy-gradient loss,
averaging with a mask, and gradient scaling.
26
The following interface is recommended:
def grpo_microbatch_train_step(
policy_log_probs: torch.Tensor,
response_mask: torch.Tensor,
gradient_accumulation_steps: int,
loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
raw_rewards: torch.Tensor | None = None,
advantages: torch.Tensor | None = None,
old_log_probs: torch.Tensor | None = None,
cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
Execute a forward-and-backward pass on a microbatch.
Args:
policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
policy being trained.
response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
prompt/padding.
gradient_accumulation_steps Number of microbatches per optimizer step.
loss_type One of "no_baseline", "reinforce_with_baseline", "grpo_clip".
raw_rewards Needed when loss_type == "no_baseline"; shape (batch_size, 1).
advantages Needed when loss_type != "no_baseline"; shape (batch_size, 1).
old_log_probs Required for GRPO-Clip; shape (batch_size, sequence_length).
cliprange Clip parameter ϵ for GRPO-Clip.
Returns:
tuple[torch.Tensor, dict[str, torch.Tensor]].
loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
this so we can log it.
metadata Dict with metadata from the underlying loss call, and any other statistics you
might want to log.
Implementation tips:
• You should call loss.backward() in this function. Make sure to adjust for gradient
accumulation.
To test your code, implement [adapters.run_grpo_microbatch_train_step]. Then run uv
run pytest -k test_grpo_microbatch_train_step and confirm it passes.

GRPO microbatch train step. Now we are ready to implement a single microbatch train step for GRPO
(recall that for a train minibatch, we iterate over many microbatches if gradient_accumulation_steps >
1).
Specifically, given the raw rewards or advantages and log probs, we will compute the per-token loss, use
masked_mean to aggregate to a scalar loss per example, average over the batch dimension, adjust for gradient
accumulation, and backpropagate.
"""


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, metadata = compute_policy_gradient_loss(policy_log_probs,loss_type,raw_rewards,advantages,old_log_probs,cliprange)
    loss = masked_mean(loss,response_mask,dim=-1)
    loss = loss.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, metadata


"""Putting it all together: GRPO train loop. Now we will put together a complete train loop for
GRPO. You should refer to the algorithm in Section 7.1 for the overall structure, using the methods we’ve
implemented where appropriate.
Below we provide some starter hyperparameters. If you have a correct implementation, you should see
reasonable results with these.
n_grpo_steps: int = 200
learning_rate: float = 1e-5
27
advantage_eps: float = 1e-6
rollout_batch_size: int = 256
group_size: int = 8
sampling_temperature: float = 1.0
sampling_min_tokens: int = 4 # As in Expiter, disallow empty string responses
sampling_max_tokens: int = 1024
epochs_per_rollout_batch: int = 1 # On-policy
train_batch_size: int = 256 # On-policy
gradient_accumulation_steps: int = 128 # microbatch size is 2, will fit on H100
gpu_memory_utilization: float = 0.85
loss_type: Literal[
"no_baseline",
"reinforce_with_baseline",
"grpo_clip",
] = "reinforce_with_baseline"
use_std_normalization: bool = True
optimizer = torch.optim.AdamW(
policy.parameters(),
lr=learning_rate,
weight_decay=0.0,
betas=(0.9, 0.95),
)
These default hyperparameters will start you in the on-policy setting—for each rollout batch, we take a
single gradient step. In terms of hyperparameters, this means that train_batch_size is equal to rollout_ ⌋
batch_size, and epochs_per_rollout_batch is equal to 1.
Here are some sanity check asserts and constants that should remove some edge cases and point you in
the right direction:
assert train_batch_size % gradient_accumulation_steps == 0, (
"train_batch_size must be divisible by gradient_accumulation_steps"
)
micro_train_batch_size = train_batch_size // gradient_accumulation_steps
assert rollout_batch_size % group_size == 0, (
"rollout_batch_size must be divisible by group_size"
)
n_prompts_per_rollout_batch = rollout_batch_size // group_size
assert train_batch_size >= group_size, (
"train_batch_size must be greater than or equal to group_size"
)
n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
And here are a few additional tips:
• Remember to use the r1_zero prompt, and direct vLLM to stop generation at the second answer tag
</answer>, as in the previous experiments.
• We suggest using typer for argument parsing.
• Use gradient clipping with clip value 1.0.
• You should routinely log validation rewards (e.g., every 5 or 10 steps). You should evaluate on at least
1024 validation examples to compare hyperparameters, as CoT/RL evaluations can be noisy.
• With our implementation of the losses, GRPO-Clip should only be used when off-policy (since it
requires the old log-probabilities).
• In the off-policy setting with multiple epochs of gradient updates per rollout batch, it would be wasteful
to recompute the old log-probabilities for each epoch. Instead, we can compute the old log-probabilities
28
once and reuse them for each epoch.
• You should not differentiate with respect to the old log-probabilities.
• You should log some or all of the following for each optimizer update:
– The loss.
– Gradient norm.
– Token entropy.
– Clip fraction, if off-policy.
– Train rewards (total, format, and answer).
– Anything else you think could be useful for debugging.
Problem (grpo_train_loop): GRPO train loop (5 points)
Deliverable: Implement a complete train loop for GRPO. Begin training a policy on MATH and
confirm that you see validation rewards improving, along with sensible rollouts over time. Provide a
plot with the validation rewards with respect to steps, and a few example rollouts over time."""


import re
import math

def extract_answer_from_response(response):
    """
    从模型响应中提取答案
    
    Args:
        response: 模型的响应文本
    
    Returns:
        str: 提取的答案，如果未找到则返回None
    """
    # 首先尝试从<answer>标签中提取
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 如果没有找到<answer>标签，尝试从####格式中提取（GSM8K格式）
    gsm8k_pattern = r'####\s*([^\n]+)'
    match = re.search(gsm8k_pattern, response)
    if match:
        return match.group(1).strip()
    
    # 尝试提取最后一个数字
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if numbers:
        return numbers[-1]
    
    return None

def extract_answer_from_truth(truth):
    """
    从真实答案中提取数值答案
    
    Args:
        truth: 真实答案文本
    
    Returns:
        str: 提取的数值答案
    """
    # 尝试从####格式中提取（GSM8K格式）
    gsm8k_pattern = r'####\s*([^\n]+)'
    match = re.search(gsm8k_pattern, truth)
    if match:
        return match.group(1).strip()
    
    # 如果没有找到，尝试提取最后一个数字
    numbers = re.findall(r'-?\d+\.?\d*', truth)
    if numbers:
        return numbers[-1]
    
    return truth.strip()

def normalize_number(num_str):
    """
    标准化数字字符串，处理分数、小数等格式
    
    Args:
        num_str: 数字字符串
    
    Returns:
        float: 标准化后的数字，如果无法解析则返回None
    """
    if num_str is None:
        return None
    
    # 移除空格
    num_str = num_str.strip()
    
    # 处理分数格式 (如 "1/2", "3/4")
    if '/' in num_str:
        try:
            parts = num_str.split('/')
            if len(parts) == 2:
                numerator = float(parts[0].strip())
                denominator = float(parts[1].strip())
                if denominator != 0:
                    return numerator / denominator
        except (ValueError, ZeroDivisionError):
            pass
    
    # 处理普通数字
    try:
        return float(num_str)
    except ValueError:
        return None

def reward_fn(response, truth):
    """
    计算奖励函数，包括格式奖励和答案奖励
    
    Args:
        response: 模型的响应
        truth: 真实答案
    
    Returns:
        dict: 包含reward, format_reward, answer_reward的字典
    """
    rewards_dict = {}
    
    # 提取答案
    response_answer = extract_answer_from_response(response)
    truth_answer = extract_answer_from_truth(truth)
    
    # 计算格式奖励
    format_reward = 0.0
    if '<answer>' in response and '</answer>' in response:
        format_reward = 1.0
    elif response_answer is not None:
        # 即使没有<answer>标签，如果能提取到答案也给部分格式奖励
        format_reward = 0.5
    
    # 计算答案奖励
    answer_reward = 0.0
    if response_answer is not None and truth_answer is not None:
        # 标准化数字
        response_num = normalize_number(response_answer)
        truth_num = normalize_number(truth_answer)
        
        if response_num is not None and truth_num is not None:
            # 检查答案是否匹配
            if abs(response_num - truth_num) < 1e-6:  # 使用小的容差处理浮点数精度问题
                answer_reward = 1.0
            else:
                # 如果答案不匹配，根据数值接近程度给部分奖励
                if truth_num != 0:
                    relative_error = abs(response_num - truth_num) / abs(truth_num)
                    if relative_error < 0.1:  # 误差在10%以内
                        answer_reward = 0.5
                    elif relative_error < 0.5:  # 误差在50%以内
                        answer_reward = 0.2
    
    # 计算总奖励
    total_reward = format_reward + answer_reward
    
    rewards_dict['reward'] = total_reward
    rewards_dict['format_reward'] = format_reward
    rewards_dict['answer_reward'] = answer_reward
    
    # 只在有奖励时打印关键信息
    if total_reward > 0:
        print(f"Reward: {total_reward:.3f} (format: {format_reward:.3f}, answer: {answer_reward:.3f})")
    
    return rewards_dict


def check_gpu_memory():
    """
    检查GPU显存使用情况
    """
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            allocated_memory = torch.cuda.memory_allocated(i) / 1024**3  # GB
            cached_memory = torch.cuda.memory_reserved(i) / 1024**3  # GB
            free_memory = total_memory - allocated_memory
            
            print(f"GPU {i}: {allocated_memory:.1f}GB/{total_memory:.1f}GB ({allocated_memory/total_memory*100:.1f}%)")
    else:
        print("No CUDA devices found")


def test_reward_function():
    """
    测试奖励函数的功能
    """
    print("Testing reward function...")
    
    # 测试用例1: 完美格式和正确答案
    response1 = "Let me solve this step by step. First, I need to calculate... The answer is <answer>72</answer>"
    truth1 = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
    result1 = reward_fn(response1, truth1)
    print(f"Test 1 (perfect): {result1}")
    
    # 测试用例2: 格式正确但答案错误
    response2 = "The calculation shows that the answer is <answer>100</answer>"
    truth2 = "#### 72"
    result2 = reward_fn(response2, truth2)
    print(f"Test 2 (format correct, answer wrong): {result2}")
    
    # 测试用例3: 没有格式标签但答案正确
    response3 = "After calculating, I get 72 as the final answer."
    truth3 = "#### 72"
    result3 = reward_fn(response3, truth3)
    print(f"Test 3 (no format, answer correct): {result3}")
    
    # 测试用例4: 完全错误的响应
    response4 = "I don't know how to solve this problem."
    truth4 = "#### 72"
    result4 = reward_fn(response4, truth4)
    print(f"Test 4 (completely wrong): {result4}")
    
    # 测试用例5: 分数答案
    response5 = "The answer is <answer>1/2</answer>"
    truth5 = "#### 0.5"
    result5 = reward_fn(response5, truth5)
    print(f"Test 5 (fraction): {result5}")
    
    print("Reward function test completed!")





def init_vllm(model_id: str, gpu_memory_utilization: float = 0.85):
    """
    初始化vLLM模型
    """
    return LLM(
        model=model_id,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=torch.bfloat16,
        enable_prefix_caching=True,
    )


def load_policy_into_vllm_instance(policy, llm):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())





def generate_model_response_vllm(vllm_model, prompt_strs, max_new_tokens=100, temperature=0.7):
    """
    使用vLLM模型生成文本响应
    
    Args:
        vllm_model: vLLM模型实例
        prompt_strs: 提示词列表
        max_new_tokens: 最大生成token数
        temperature: 采样温度
    
    Returns:
        list[str]: 生成的响应文本列表
    """
    from vllm import SamplingParams
    
    # 创建采样参数对象
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_new_tokens,
        stop=["\n"]  # 在新行处停止生成
    )
    
    try:
        # 生成文本
        outputs = vllm_model.generate(prompt_strs, sampling_params)
        
        # 提取生成的文本
        responses = []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            responses.append(generated_text)
        
        return responses
        
    except Exception as e:
        print(f"vLLM generation failed: {e}")
        # 如果vLLM生成失败，返回默认响应
        return ["Generation failed"] * len(prompt_strs)








def grpo_train_loop(cfg):
    n_grpo_steps = cfg['n_grpo_steps']
    learning_rate = cfg['learning_rate']
    advantage_eps = cfg['advantage_eps']
    rollout_batch_size = cfg['rollout_batch_size']
    group_size = cfg['group_size']
    sampling_temperature = cfg['sampling_temperature']
    sampling_min_tokens = cfg['sampling_min_tokens']
    sampling_max_tokens = cfg['sampling_max_tokens']
    epochs_per_rollout_batch = cfg['epochs_per_rollout_batch']
    train_batch_size = cfg['train_batch_size']
    gradient_accumulation_steps = cfg['gradient_accumulation_steps']
    loss_type = cfg['loss_type']
    use_std_normalization = cfg['use_std_normalization']
    train_data_arr = load_train_data()
    test_data_arr = load_test_data()
    test_prompt_strs, test_output_strs = sample_data(test_data_arr)
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-1.5B')
    
    # 双GPU配置：GPU 0用于推理，GPU 1用于训练
    train_device = "cuda:1"  # 训练使用GPU 1
    inference_device = "cuda:0"  # 推理使用GPU 0
    
    print(f"Training on {train_device}, inference on {inference_device}")
    print("Using vLLM for inference")
    
    # 检查初始显存状态
    print("Initial GPU memory:")
    check_gpu_memory()
    
    # 加载训练模型到GPU 1
    print(f"Loading training model to {train_device}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Math-1.5B",
            torch_dtype=torch.float16,  # 使用半精度以节省显存
            device_map="cuda:1",  # 指定使用GPU 1
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✓ Training model loaded successfully")
    except Exception as e:
        print(f"Training model loading failed: {e}")
        raise
    
    # 初始化vLLM推理模型
    print(f"Initializing vLLM inference model...")
    try:
        vllm_model = init_vllm(
            model_id="Qwen/Qwen2.5-Math-1.5B",
            gpu_memory_utilization=0.85
        )
        print("✓ vLLM inference model initialized successfully")
    except Exception as e:
        print(f"vLLM initialization failed: {e}")
        raise
    
    # 确保模型在正确的设备上
    device = next(model.parameters()).device
    print(f"Training model loaded on device: {device}")
    
    # 检查模型加载后的显存状态
    print("GPU memory after model loading:")
    check_gpu_memory()
    
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=0.0,betas=(0.9,0.95))
    
    # 初始同步vLLM推理模型权重
    print("Initial vLLM weight synchronization...")
    try:
        load_policy_into_vllm_instance(model, vllm_model)
        print("✓ vLLM weights synced successfully")
    except Exception as e:
        print(f"vLLM weight sync failed: {e}")
    
    # 设置权重同步频率
    sync_frequency = 5  # 每5步同步一次权重
    
    for step in range(n_grpo_steps):
        prompt_strs, output_strs = sample_data(train_data_arr,8)
        res = tokenize_prompt_and_output(prompt_strs,output_strs,tokenizer,device)
        policy_log_probs = get_response_log_probs(model,res['input_ids'],res['labels'],return_token_entropy=True)['log_probs']
        old_log_probs = policy_log_probs.detach().clone()
        response_mask = res['response_mask']
        # 使用vLLM进行推理
        responses = generate_model_response_vllm(vllm_model, prompt_strs, max_new_tokens=sampling_max_tokens, temperature=sampling_temperature)
        rewards_normalized, rewards, metadata = compute_group_normalized_rewards(reward_fn,responses,output_strs,group_size,advantage_eps,use_std_normalization,device)
        for _ in range(epochs_per_rollout_batch):
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            policy_log_probs = get_response_log_probs(model,res['input_ids'],res['labels'],return_token_entropy=True)['log_probs']  # 双GPU可以计算entropy
            optimizer.zero_grad()
            raw_rewards = metadata['raw_rewards']
            advantages = rewards_normalized
            cliprange = cfg['cliprange']

            # 只打印关键数据
            print(f'Step {step}: advantages=[{advantages.min().item():.3f}, {advantages.max().item():.3f}], '
                  f'log_probs=[{policy_log_probs.min().item():.3f}, {policy_log_probs.max().item():.3f}]')
            
            # 检查输入数据是否包含NaN
            if torch.isnan(policy_log_probs).any():
                print(f"WARNING: policy_log_probs contains NaN before backward")
                continue
            if torch.isnan(advantages).any():
                print(f"WARNING: advantages contains NaN before backward")
                continue
            if torch.isnan(old_log_probs).any():
                print(f"WARNING: old_log_probs contains NaN before backward")
                continue
                
            loss, metadata = grpo_microbatch_train_step(policy_log_probs,response_mask,gradient_accumulation_steps,loss_type,raw_rewards,advantages,old_log_probs,cliprange)
            
            # 检查损失值是否为NaN或无穷大
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: Loss is {loss.item()}, skipping step")
                continue
            
            # 检查所有参数的梯度是否包含NaN或Inf
            nan_grad_count = 0
            inf_grad_count = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        nan_grad_count += 1
                        print(f"WARNING: Gradient for {name} contains NaN")
                    if torch.isinf(param.grad).any():
                        inf_grad_count += 1
                        print(f"WARNING: Gradient for {name} contains Inf")
            
            if nan_grad_count > 0 or inf_grad_count > 0:
                print(f"WARNING: Found {nan_grad_count} NaN gradients and {inf_grad_count} Inf gradients, skipping step")
                continue
            
            # 检查optimizer.step()前后某个参数的变化
            # 选择一个代表性的参数进行监控
            sample_param = None
            sample_param_name = None
            for name, param in model.named_parameters():
                if 'embed_tokens.weight' in name:  # 选择embedding层作为监控对象
                    sample_param = param
                    sample_param_name = name
                    break
            
            if sample_param is not None:
                # 记录更新前的参数值
                param_before = sample_param.data.clone()
                param_mean_before = param_before.mean().item()
                param_std_before = param_before.std().item()
                param_min_before = param_before.min().item()
                param_max_before = param_before.max().item()
                
                # 记录梯度信息
                if sample_param.grad is not None:
                    # 检查梯度是否包含NaN或Inf
                    grad_has_nan = torch.isnan(sample_param.grad).any().item()
                    grad_has_inf = torch.isinf(sample_param.grad).any().item()
                    
                    if grad_has_nan or grad_has_inf:
                        print(f"WARNING: Gradient for {sample_param_name} contains NaN: {grad_has_nan}, Inf: {grad_has_inf}")
                        # 如果梯度包含NaN，跳过此步骤
                        continue
                    
                    grad_mean = sample_param.grad.mean().item()
                    grad_std = sample_param.grad.std().item()
                    grad_norm = sample_param.grad.norm().item()
                    grad_min = sample_param.grad.min().item()
                    grad_max = sample_param.grad.max().item()
                    
                    print(f"Before step - {sample_param_name}:")
                    print(f"  Param: mean={param_mean_before:.6f}, std={param_std_before:.6f}, range=[{param_min_before:.6f}, {param_max_before:.6f}]")
                    print(f"  Grad: mean={grad_mean:.6f}, std={grad_std:.6f}, norm={grad_norm:.6f}, range=[{grad_min:.6f}, {grad_max:.6f}]")
                else:
                    print(f"Before step - {sample_param_name}: No gradient")
            
            optimizer.step()
            print(f'Loss: {loss.item():.6f}')
            
            # 检查更新后的参数值
            if sample_param is not None:
                param_after = sample_param.data
                param_mean_after = param_after.mean().item()
                param_std_after = param_after.std().item()
                param_min_after = param_after.min().item()
                param_max_after = param_after.max().item()
                
                # 计算参数变化
                param_change = param_after - param_before
                change_mean = param_change.mean().item()
                change_std = param_change.std().item()
                change_norm = param_change.norm().item()
                
                print(f"After step - {sample_param_name}:")
                print(f"  Param: mean={param_mean_after:.6f}, std={param_std_after:.6f}, range=[{param_min_after:.6f}, {param_max_after:.6f}]")
                print(f"  Change: mean={change_mean:.6f}, std={change_std:.6f}, norm={change_norm:.6f}")
                
                # 检查参数是否包含NaN
                if torch.isnan(param_after).any():
                    print(f"WARNING: {sample_param_name} contains NaN after optimizer.step()")
                if torch.isinf(param_after).any():
                    print(f"WARNING: {sample_param_name} contains Inf after optimizer.step()")
            
            # 检查模型权重是否包含NaN
            nan_params = 0
            total_params = 0
            for name, param in model.named_parameters():
                total_params += 1
                if torch.isnan(param).any():
                    nan_params += 1
                    print(f"WARNING: Parameter {name} contains NaN")
            
            if nan_params > 0:
                print(f"WARNING: {nan_params}/{total_params} parameters contain NaN, stopping training")
                break
            
            # 根据频率同步vLLM推理模型权重
            if (step + 1) % sync_frequency == 0:
                print(f"Syncing vLLM weights at step {step}...")
                try:
                    load_policy_into_vllm_instance(model, vllm_model)
                    print("✓ vLLM weights synced successfully")
                except Exception as e:
                    print(f"vLLM weight sync failed: {e}")
            
            # 训练后清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 训练结束后清理资源
    print("Training completed, cleaning up resources...")
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Resource cleanup completed")


config = {
    'n_grpo_steps': 100,
    'learning_rate': 1e-6,  # 进一步降低学习率
    'advantage_eps': 1e-6,
    'rollout_batch_size': 128,  # 双GPU可以支持更大的批次
    'group_size': 8,  # 恢复原始组大小
    'sampling_temperature': 1.0,
    'sampling_min_tokens': 4,
    'sampling_max_tokens': 1024,  # 恢复原始生成长度
    'epochs_per_rollout_batch': 1,
    'train_batch_size': 128,  # 双GPU可以支持更大的训练批次
    'gradient_accumulation_steps': 8,  # 降低梯度累积步数
    'loss_type': 'reinforce_with_baseline',
    'use_std_normalization': True,
    'cliprange': 0.2
}

# 取消注释下面的行来测试奖励函数
# test_reward_function()

# 开始训练
if __name__ == '__main__':
    grpo_train_loop(config)
