from cs336_alignment.vllm_helper import init_vllm, load_policy_into_vllm_instance
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.grpo import compute_group_normalized_rewards, grpo_microbatch_train_step
from cs336_alignment.train_grpo import grpo_microbatch_train_step, tokenize_prompt_and_output, sample_dataset, duplicate_data
from cs336_alignment.sft_helper import get_response_log_probs
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import random
import argparse
import time

from rl import load_train_data, load_test_data, sample_data, generate_model_response_vllm


def grpo_train_loop(cfg):
    # Extract training parameters
    training_params = {
        'n_grpo_steps': cfg['n_grpo_steps'],
        'learning_rate': cfg['learning_rate'],
        'advantage_eps': cfg['advantage_eps'],
        'rollout_batch_size': cfg['rollout_batch_size'],
        'group_size': cfg['group_size'],
        'sampling_temperature': cfg['sampling_temperature'],
        'sampling_min_tokens': cfg['sampling_min_tokens'],
        'sampling_max_tokens': cfg['sampling_max_tokens'],
        'epochs_per_rollout_batch': cfg['epochs_per_rollout_batch'],
        'train_batch_size': cfg['train_batch_size'],
        'gradient_accumulation_steps': cfg['gradient_accumulation_steps'],
        'loss_type': cfg['loss_type'],
        'use_std_normalization': cfg['use_std_normalization'],
        'eval_log_frequency': cfg.get('eval_log_frequency', 5),
        'eval_sample_size': cfg.get('eval_sample_size', 64),
    }
    
    # Load data
    train_data_arr = load_train_data()
    test_data_arr = load_test_data()
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-1.5B')
    
    # Add assertions from reference implementation
    assert training_params['train_batch_size'] % training_params['gradient_accumulation_steps'] == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = training_params['train_batch_size'] // training_params['gradient_accumulation_steps']

    assert training_params['rollout_batch_size'] % training_params['group_size'] == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = training_params['rollout_batch_size'] // training_params['group_size']

    assert training_params['train_batch_size'] >= training_params['group_size'], (
        "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = training_params['rollout_batch_size'] // micro_train_batch_size

    # 双GPU配置：GPU 0用于推理，GPU 1用于训练
    train_device = "cuda:1"  # 训练使用GPU 1
    inference_device = "cuda:0"  # 推理使用GPU 0
    
    print(f"Training on {train_device}, inference on {inference_device}")
    print("Using vLLM for inference")
    
    # 加载训练模型到GPU 1
    print(f"Loading training model to {train_device}...")
    try:
        policy = AutoModelForCausalLM.from_pretrained(
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
        vllm = init_vllm(
            model_id="Qwen/Qwen2.5-Math-1.5B",
            gpu_memory_utilization=0.85,
            device='cuda:0',
            seed=42
        )
        print("✓ vLLM inference model initialized successfully")
    except Exception as e:
        print(f"vLLM initialization failed: {e}")
        raise
    
    # 确保模型在正确的设备上
    device = next(policy.parameters()).device
    print(f"Training model loaded on device: {device}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(policy.parameters(), lr=training_params['learning_rate'], weight_decay=0.1, betas=(0.9, 0.99))
    training_params['optimizer'] = optimizer
    
    # Initialize sampling parameters
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=training_params['sampling_temperature'],
        max_tokens=training_params['sampling_max_tokens'],
        min_tokens=training_params['sampling_min_tokens'],
        stop=["<|im_end|>", "<|endoftext|>"]
    )
    
    # 初始同步vLLM推理模型权重
    print("Initial vLLM weight synchronization...")
    try:
        load_policy_into_vllm_instance(policy, vllm)
        print("✓ vLLM weights synced successfully")
    except Exception as e:
        print(f"vLLM weight sync failed: {e}")
    
    # Setup output directory
    output_dir = cfg.get('output_dir', './output')
    experiment_name = cfg.get('experiment_name', 'grpo_experiment')
    
    # Directory to store all models
    model_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    # Initialize logging
    print(f"Starting GRPO training experiment: {experiment_name}")
    print(f"Training parameters: {training_params}")
    print(f"Output directory: {output_dir}")
    print(f"Model directory: {model_dir}")
    print("=" * 80)

    train_step = 0
    eval_step = 0
    start_time = time.time()

    for grpo_step_idx in range(training_params['n_grpo_steps']):
        load_policy_into_vllm_instance(policy, vllm)

        # Evaluation logic
        if grpo_step_idx % training_params['eval_log_frequency'] == (training_params['eval_log_frequency'] - 1):
            sampled_eval_data = sample_dataset(test_data_arr, training_params['eval_sample_size'])
            prompts_batch = sampled_eval_data['prompts']
            answers_batch = sampled_eval_data['answers']

            vllm_rollouts = vllm.generate(prompts_batch, sampling_params)

            rollout_input_text = []
            rollout_response_text = []

            for rollout in vllm_rollouts:
                for r in rollout.outputs:
                    rollout_input_text.append(rollout.prompt)
                    rollout_response_text.append(r.text)
            
            _, _, reward_metadata = compute_group_normalized_rewards(
                r1_zero_reward_fn,
                rollout_response_text,
                answers_batch,
                1,
                training_params['advantage_eps'],
                training_params['use_std_normalization'],
            )

            # Print a randomly sampled eval response
            eval_rand_idx = random.randrange(training_params['eval_sample_size'])
            print(f'\n=== EVALUATION STEP {eval_step} ===')
            print(f'GRPO Step: {grpo_step_idx}')
            print(f'Evaluation Accuracy: {reward_metadata["mean"]:.4f}')
            print(f'Reward Metadata: {reward_metadata}')
            print('\nSample Response:')
            print(f'Prompt: {rollout_input_text[eval_rand_idx]}')
            print(f'Correct Answer: {answers_batch[eval_rand_idx]}')
            print(f'LLM Response: {rollout_response_text[eval_rand_idx]}')
            print('=' * 50)

            # Save model
            curr_model_dir = os.path.join(model_dir, 'eval_step_{}'.format(eval_step))
            policy.save_pretrained(save_directory=curr_model_dir)
            tokenizer.save_pretrained(save_directory=curr_model_dir)

            eval_step += 1

        # Training logic - One policy gradient step per train_batch_size of data
        for rollout_batch_idx in range(0, training_params['train_batch_size'], training_params['rollout_batch_size']):
            # Sample a batch of data, then select microbatches later
            sampled_training_data = sample_dataset(train_data_arr, n_prompts_per_rollout_batch)
            prompts_batch = sampled_training_data['prompts']
            answers_batch = sampled_training_data['answers']

            prompts_batch = duplicate_data(prompts_batch, training_params['group_size'])
            answers_batch = duplicate_data(answers_batch, training_params['group_size'])

            vllm_rollouts = vllm.generate(prompts_batch, sampling_params)

            rollout_input_text = []
            rollout_response_text = []

            for rollout in vllm_rollouts:
                for r in rollout.outputs:
                    rollout_input_text.append(rollout.prompt)
                    rollout_response_text.append(r.text)
            
            advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
                r1_zero_reward_fn,
                rollout_response_text,
                answers_batch,
                training_params['group_size'],
                training_params['advantage_eps'],
                training_params['use_std_normalization'],
            )

            # Print training progress
            if train_step % 10 == 0:  # Print every 10 training steps
                print(f'\n--- Training Step {train_step} ---')
                print(f'GRPO Step: {grpo_step_idx}, Rollout Batch: {rollout_batch_idx}')
                print(f'Reward Mean: {reward_metadata["mean"]:.4f}')
                print(f'Reward Std: {reward_metadata.get("std", 0):.4f}')
                print(f'Advantages - Mean: {advantages.mean().item():.4f}, Std: {advantages.std().item():.4f}')
                print(f'Raw Rewards - Mean: {raw_rewards.mean().item():.4f}, Std: {raw_rewards.std().item():.4f}')

            rollout_data_tokenized = tokenize_prompt_and_output(
                rollout_input_text,
                rollout_response_text,
                tokenizer
            )

            for _ in range(training_params['epochs_per_rollout_batch']):
                training_params['optimizer'].zero_grad()

                rollout_batch_loss = 0

                for microbatch_idx in range(n_microbatches_per_rollout_batch):
                    microbatch_slice = slice(
                        microbatch_idx * micro_train_batch_size,
                        (microbatch_idx + 1) * micro_train_batch_size
                    )

                    microbatch_input_ids = rollout_data_tokenized['input_ids'][microbatch_slice].to(device)
                    microbatch_labels = rollout_data_tokenized['labels'][microbatch_slice].to(device)
                    microbatch_response_mask = rollout_data_tokenized['response_mask'][microbatch_slice].to(device)

                    advantages_microbatch = advantages[microbatch_slice].to(device)
                    raw_rewards_microbatch = raw_rewards[microbatch_slice].to(device)

                    policy_log_probs_dict = get_response_log_probs(
                        policy,
                        microbatch_input_ids,
                        microbatch_labels,
                        return_token_entropy=True
                    )
                    policy_log_probs = policy_log_probs_dict['log_probs']
                    policy_token_entropy = policy_log_probs_dict['token_entropy']

                    old_log_probs = policy_log_probs # change this when doing off-policy updates

                    advantages_microbatch = advantages_microbatch.unsqueeze(-1)

                    loss, loss_metadata = grpo_microbatch_train_step(
                        policy_log_probs,
                        microbatch_response_mask,
                        training_params['gradient_accumulation_steps'],
                        training_params['loss_type'],
                        raw_rewards_microbatch,
                        advantages_microbatch,
                        old_log_probs,
                        1.0,
                    )

                    rollout_batch_loss += loss.item()
            
                training_params['optimizer'].step()

                rollout_batch_loss /= n_microbatches_per_rollout_batch
                
                # Print loss information
                if train_step % 5 == 0:  # Print every 5 training steps
                    print(f'  Loss: {loss.item():.6f}, Avg Batch Loss: {rollout_batch_loss:.6f}')
                
                train_step += 1
    
    # Training completed
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f'\n=== TRAINING COMPLETED ===')
    print(f'Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)')
    print(f'Total GRPO steps: {training_params["n_grpo_steps"]}')
    print(f'Total training steps: {train_step}')
    print(f'Total evaluation steps: {eval_step}')

    # Save final model
    model_final_dir = os.path.join(model_dir, 'final')
    print(f'Saving final model to: {model_final_dir}')
    policy.save_pretrained(save_directory=model_final_dir)
    tokenizer.save_pretrained(save_directory=model_final_dir)
    
    print('Training complete!')


config = {
    'n_grpo_steps': 500,
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
    'cliprange': 0.2,
    'eval_log_frequency': 5,
    'eval_sample_size': 64,
    'output_dir': './output',
    'experiment_name': 'grpo_experiment_no_wandb'
}

# 取消注释下面的行来测试奖励函数
# test_reward_function()

# 开始训练
if __name__ == '__main__':
    grpo_train_loop(config)
