from rl import *
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


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
    
    # 手动实现AdamW优化器状态
    optimizer_state = {}
    for name, param in model.named_parameters():
        optimizer_state[name] = {
            'exp_avg': torch.zeros_like(param.data),  # 一阶矩估计
            'exp_avg_sq': torch.zeros_like(param.data),  # 二阶矩估计
            'step': 0
        }
    
    # AdamW参数
    beta1, beta2 = 0.9, 0.99
    weight_decay = 0.1
    eps = 1e-8
    
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
        
        # 每隔几轮打印responses例子
        if step % 5 == 0:  # 每5步打印一次
            print(f"\n=== Step {step} - Sample Responses ===")
            for i in range(min(1, len(prompt_strs))):  # 打印前3个例子
                print(f"\nExample {i+1}:")
                print(f"Prompt: {prompt_strs[i]}")
                print(f"Response: {responses[i]}")
                print(f"Ground Truth: {output_strs[i]}")
            print("=" * 50)
        
        rewards_normalized, rewards, metadata = compute_group_normalized_rewards(r1_zero_reward_fn,responses,output_strs,group_size,advantage_eps,use_std_normalization,device)
        
        # 每隔几轮打印奖励统计
        if step % 5 == 0:  # 每5步打印一次
            print(f"\n=== Step {step} - Reward Statistics ===")
            print(f"Raw rewards: {rewards.flatten().tolist()}")
            print(f"Advantages: {rewards_normalized.flatten().tolist()}")
            print(f"Reward mean: {rewards.mean().item():.4f}, std: {rewards.std().item():.4f}")
            print(f"Advantage mean: {rewards_normalized.mean().item():.4f}, std: {rewards_normalized.std().item():.4f}")
            print("=" * 50)
        for _ in range(epochs_per_rollout_batch):
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            policy_log_probs = get_response_log_probs(model,res['input_ids'],res['labels'],return_token_entropy=True)['log_probs']  # 双GPU可以计算entropy
            # 手动清零梯度
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            raw_rewards = metadata['raw_rewards'].reshape(-1,1)
            advantages = rewards_normalized.reshape(-1,1)
            cliprange = cfg['cliprange']

            # 只打印关键数据
            # print('advantages: ',advantages)
            # print('policy_log_probs: ',policy_log_probs)
            # print(f'Step {step}: advantages=[{advantages.min().item():.3f}, {advantages.max().item():.3f}], '
            #       f'log_probs=[{policy_log_probs.min().item():.3f}, {policy_log_probs.max().item():.3f}]')
            
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
            # 手动更新参数，检查NaN
            print(f'Loss: {loss.item():.6f}')
            
            # 手动实现AdamW优化器步骤
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad = param.grad
                    state = optimizer_state[name]
                    state['step'] += 1
                    
                    # 检查梯度是否包含NaN/Inf
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        print(f"ERROR: Gradient for {name} contains NaN/Inf before optimization")
                        print(f"  Gradient stats: mean={grad.mean().item():.8f}, std={grad.std().item():.8f}")
                        print(f"  Gradient min/max: {grad.min().item():.8f}/{grad.max().item():.8f}")
                        break
                    
                    # 检查参数是否包含NaN/Inf
                    if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                        print(f"ERROR: Parameter {name} contains NaN/Inf before optimization")
                        print(f"  Parameter stats: mean={param.data.mean().item():.8f}, std={param.data.std().item():.8f}")
                        break
                    
                    # 应用权重衰减 (AdamW风格)
                    param.data.mul_(1 - learning_rate * weight_decay)
                    
                    # 检查权重衰减后是否出现NaN/Inf
                    if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                        print(f"ERROR: Parameter {name} contains NaN/Inf after weight decay")
                        print(f"  Weight decay factor: {1 - learning_rate * weight_decay}")
                        break
                    
                    # 更新一阶矩估计 (动量)
                    state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
                    
                    # 检查动量更新后是否出现NaN/Inf
                    if torch.isnan(state['exp_avg']).any() or torch.isinf(state['exp_avg']).any():
                        print(f"ERROR: exp_avg for {name} contains NaN/Inf after momentum update")
                        print(f"  Beta1: {beta1}, Alpha: {1 - beta1}")
                        break
                    
                    # 更新二阶矩估计 (方差)
                    state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    # 检查方差更新后是否出现NaN/Inf
                    if torch.isnan(state['exp_avg_sq']).any() or torch.isinf(state['exp_avg_sq']).any():
                        print(f"ERROR: exp_avg_sq for {name} contains NaN/Inf after variance update")
                        print(f"  Beta2: {beta2}, Alpha: {1 - beta2}")
                        break
                    
                    # 偏差修正
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    
                    # 检查偏差修正因子
                    if torch.isnan(torch.tensor(bias_correction1)) or torch.isnan(torch.tensor(bias_correction2)):
                        print(f"ERROR: Bias correction factors are NaN for {name}")
                        print(f"  bias_correction1: {bias_correction1}, bias_correction2: {bias_correction2}")
                        print(f"  step: {state['step']}")
                        break
                    
                    # 计算更新步长
                    step_size = learning_rate / bias_correction1
                    denom = (state['exp_avg_sq'] / bias_correction2).sqrt().add_(eps)
                    
                    # 检查分母是否包含NaN/Inf
                    if torch.isnan(denom).any() or torch.isinf(denom).any():
                        print(f"ERROR: Denominator contains NaN/Inf for {name}")
                        print(f"  exp_avg_sq stats: mean={state['exp_avg_sq'].mean().item():.8f}, std={state['exp_avg_sq'].std().item():.8f}")
                        print(f"  bias_correction2: {bias_correction2}")
                        print(f"  eps: {eps}")
                        break
                    
                    # 检查step_size是否有效
                    if torch.isnan(torch.tensor(step_size)) or torch.isinf(torch.tensor(step_size)):
                        print(f"ERROR: step_size is NaN/Inf for {name}")
                        print(f"  learning_rate: {learning_rate}, bias_correction1: {bias_correction1}")
                        break
                    
                    # 更新参数
                    param.data.addcdiv_(state['exp_avg'], denom, value=-step_size)
                    
                    # 检查最终参数是否包含NaN/Inf
                    if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                        print(f"ERROR: Parameter {name} contains NaN/Inf after final update")
                        print(f"  Final update components:")
                        print(f"    exp_avg: mean={state['exp_avg'].mean().item():.8f}, std={state['exp_avg'].std().item():.8f}")
                        print(f"    denom: mean={denom.mean().item():.8f}, std={denom.std().item():.8f}")
                        print(f"    step_size: {step_size}")
                        print(f"    learning_rate: {learning_rate}")
                        print(f"    step: {state['step']}")
                        break

        # 每隔50步保存模型
        if (step + 1) % 100 == 0:
            save_path = f"checkpoint_step_{step + 1}"
            print(f"Saving model checkpoint at step {step + 1} to {save_path}...")
            try:
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"✓ Model checkpoint saved to {save_path}")
            except Exception as e:
                print(f"Model checkpoint save failed: {e}")
            
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
    'cliprange': 0.2
}

# 取消注释下面的行来测试奖励函数
# test_reward_function()

# 开始训练
if __name__ == '__main__':
    grpo_train_loop(config)