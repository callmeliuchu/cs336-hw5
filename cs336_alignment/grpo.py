import torch
from typing import Literal


def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
    ):

    raw_rewards = []

    for rollout_response, gt_response in zip(rollout_responses, repeated_ground_truths):
        curr_reward = reward_fn(rollout_response, gt_response)['reward']
        raw_rewards.append(curr_reward)
    
    # Compute mean reward for each group
    raw_rewards = torch.tensor(raw_rewards)
    rewards_per_group = raw_rewards.reshape((-1, group_size))
    mean_reward_per_group = torch.mean(rewards_per_group, dim=-1, keepdim=True)

    advantage = rewards_per_group - mean_reward_per_group

    if normalize_by_std:
        std_reward_per_group = torch.std(rewards_per_group, dim=-1, keepdim=True)

        advantage /= (std_reward_per_group + advantage_eps)
    
    advantage = advantage.flatten()

    metadata = {
        'mean': torch.mean(raw_rewards),
        'std': torch.std(raw_rewards),
        'max': torch.max(raw_rewards),
        'min': torch.min(raw_rewards),
    }

    return advantage, raw_rewards, metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    ) -> torch.Tensor:

    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    pi_ratio = torch.exp(policy_log_probs - old_log_probs)
    unclipped_term = advantages * pi_ratio

    clipped_term = torch.clip(pi_ratio, min=1 - cliprange, max=1 + cliprange)
    clipped_term *= advantages

    loss = -torch.minimum(unclipped_term, clipped_term)

    metadata = {
        'token_clipped': clipped_term < unclipped_term
    }

    return loss, metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    if loss_type == 'no_baseline':
        assert raw_rewards is not None

        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == 'reinforce_with_baseline':
        assert advantages is not None

        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    elif loss_type == 'grpo_clip':
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None

        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None= None,
    ) -> torch.Tensor:

    tensor_masked = tensor * mask

    return torch.sum(tensor_masked, dim=dim) / torch.sum(mask, dim=dim)

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
    )

    loss = masked_mean(loss, response_mask)
    loss /= gradient_accumulation_steps

    loss.backward()

    return loss, metadata