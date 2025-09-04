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
            
            # 计算microbatch大小
            micro_train_batch_size = train_batch_size // gradient_accumulation_steps
            actual_batch_size = res['input_ids'].shape[0]  # 实际的数据批次大小
            n_microbatches = actual_batch_size // micro_train_batch_size
            
            print(f"Actual batch size: {actual_batch_size}, micro_train_batch_size: {micro_train_batch_size}, n_microbatches: {n_microbatches}")
            
            optimizer.zero_grad()
            total_loss = 0
            
            # 梯度累积循环
            for microbatch_idx in range(n_microbatches):
                # 计算当前microbatch的切片
                start_idx = microbatch_idx * micro_train_batch_size
                end_idx = min((microbatch_idx + 1) * micro_train_batch_size, actual_batch_size)
                
                # 获取microbatch数据
                microbatch_input_ids = res['input_ids'][start_idx:end_idx]
                microbatch_labels = res['labels'][start_idx:end_idx]
                microbatch_response_mask = response_mask[start_idx:end_idx]
                microbatch_advantages = rewards_normalized[start_idx:end_idx]
                microbatch_raw_rewards = metadata['raw_rewards'][start_idx:end_idx]
                microbatch_old_log_probs = old_log_probs[start_idx:end_idx]
                
                # 检查microbatch是否为空
                if microbatch_input_ids.shape[0] == 0:
                    print(f"WARNING: microbatch {microbatch_idx} is empty, skipping")
                    continue
                
                # 检查microbatch数据是否包含NaN
                if torch.isnan(microbatch_input_ids).any() or torch.isnan(microbatch_labels).any():
                    print(f"WARNING: microbatch {microbatch_idx} input data contains NaN")
                    continue
                if torch.isnan(microbatch_advantages).any():
                    print(f"WARNING: microbatch {microbatch_idx} advantages contains NaN")
                    continue
                if torch.isnan(microbatch_old_log_probs).any():
                    print(f"WARNING: microbatch {microbatch_idx} old_log_probs contains NaN")
                    continue
                
                # 计算当前microbatch的log_probs
                print('microbatch_input_ids',microbatch_input_ids.shape)
                print('microbatch_labels',microbatch_labels.shape)
                policy_log_probs = get_response_log_probs(model, microbatch_input_ids, microbatch_labels, return_token_entropy=True)['log_probs']
                
                if torch.isnan(policy_log_probs).any():
                    print(f"WARNING: microbatch {microbatch_idx} policy_log_probs contains NaN")
                    continue
                
                # 执行microbatch训练步骤
                print('start end',start_idx,end_idx)
                print('input_ids',res['input_ids'].shape)
                print('labels',res['labels'].shape)
                print('response_mask',res['response_mask'].shape)
                print('policy_log_probs',policy_log_probs.shape)
                print('microbatch_response_mask',microbatch_response_mask.shape)
                print('gradient_accumulation_steps',gradient_accumulation_steps)
                print('loss_type',loss_type)
                print('microbatch_raw_rewards',microbatch_raw_rewards.shape)
                print('microbatch_advantages',microbatch_advantages.shape)
                print('microbatch_old_log_probs',microbatch_old_log_probs.shape)
                loss, loss_metadata = grpo_microbatch_train_step(
                    policy_log_probs,
                    microbatch_response_mask,
                    gradient_accumulation_steps,
                    loss_type,
                    microbatch_raw_rewards.reshape(-1,1),
                    microbatch_advantages.reshape(-1,1),
                    microbatch_old_log_probs,
                    cfg['cliprange']
                )
                
                total_loss += loss.item()
            
            # 所有microbatch处理完成后，执行优化器步骤
            if n_microbatches > 0:
                avg_loss = total_loss / n_microbatches
                print(f'Average Loss: {avg_loss:.6f} (over {n_microbatches} microbatches)')
                
                # 添加梯度裁剪以防止NaN
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 检查梯度是否包含NaN
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"WARNING: Gradient for {name} contains NaN")
                        has_nan_grad = True
                        break
                
                if not has_nan_grad:
                    # 手动更新参数，避免optimizer.step()的NaN问题
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            # 简单的SGD更新: param = param - lr * grad
                            param.data = param.data - learning_rate * param.grad
                            
                            # 检查更新后的参数是否包含NaN
                            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                                print(f"ERROR: Parameter {name} became NaN/Inf after manual update")
                                print(f"  Learning rate: {learning_rate}")
                                print(f"  Gradient stats: mean={param.grad.mean().item():.8f}, std={param.grad.std().item():.8f}")
                                print(f"  Parameter stats before: mean={param.data.mean().item():.8f}, std={param.data.std().item():.8f}")
                                return
                    
                    print(f"Manual parameter update completed with lr={learning_rate}")
                else:
                    print("Skipping parameter update due to NaN gradients")


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
    'learning_rate': 5e-7,  # 进一步降低学习率以提高稳定性
    'advantage_eps': 1e-6,
    'rollout_batch_size': 128,  # 双GPU可以支持更大的批次
    'group_size': 8,  # 恢复原始组大小
    'sampling_temperature': 1.0,
    'sampling_min_tokens': 4,
    'sampling_max_tokens': 1024,  # 恢复原始生成长度
    'epochs_per_rollout_batch': 1,
    'train_batch_size': 128,  # 双GPU可以支持更大的训练批次
    'gradient_accumulation_steps': 16,  # 增加梯度累积步数以提高稳定性
    'loss_type': 'reinforce_with_baseline',
    'use_std_normalization': True,
    'cliprange': 0.2
}

# 取消注释下面的行来测试奖励函数
# test_reward_function()

# 开始训练
if __name__ == '__main__':
    grpo_train_loop(config)
