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
import json
import torch
import random
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from rl import tokenize_prompt_and_output,get_response_log_probs,sft_microbatch_train_step,load_policy_into_vllm_instance,init_vllm


R1_ZERO_PROMPT = open('cs336_alignment/prompts/r1_zero.prompt', 'r').read()

def tokenize_prompt_and_output_with_limit(prompt_strs, output_strs, tokenizer, device=None, max_length=2048):
    """Tokenize with sequence length limit and monitoring"""
    prompt_tokens = tokenizer(prompt_strs, truncation=True, max_length=max_length//2)['input_ids']
    output_tokens = tokenizer(output_strs, truncation=True, max_length=max_length//2)['input_ids']

    batch_sz = len(prompt_tokens)
    prompt_and_output_lens = [len(p) + len(o) for p, o in zip(prompt_tokens, output_tokens)]
    
    # 监控序列长度
    max_seq_len = max(prompt_and_output_lens)
    avg_seq_len = sum(prompt_and_output_lens) / len(prompt_and_output_lens)
    print(f"Batch sequence lengths - Max: {max_seq_len}, Avg: {avg_seq_len:.1f}")
    
    # 如果超过限制，进一步截断
    if max_seq_len > max_length:
        print(f"⚠️  Sequence length {max_seq_len} exceeds limit {max_length}, truncating...")
        # 重新截断到更短的长度
        prompt_tokens = tokenizer(prompt_strs, truncation=True, max_length=max_length//3)['input_ids']
        output_tokens = tokenizer(output_strs, truncation=True, max_length=max_length//3)['input_ids']
        prompt_and_output_lens = [len(p) + len(o) for p, o in zip(prompt_tokens, output_tokens)]
        max_seq_len = max(prompt_and_output_lens)
        print(f"After truncation - Max: {max_seq_len}, Avg: {sum(prompt_and_output_lens)/len(prompt_and_output_lens):.1f}")
    
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
        p_o_concat_padded = torch.nn.functional.pad(p_o_concat, (0, padded_len - concat_len), 'constant', tokenizer.eos_token_id)

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

def load_math_data(data_path):
    """Load MATH dataset from jsonl file"""
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def evaluate_with_vllm(val_prompts, val_answers, epoch, policy_model, llm):
    """Evaluate model using VLLM on CUDA 1"""
    try:
        from vllm import SamplingParams
        import re
        
        # Set model to evaluation mode
        policy_model.eval()
        
        # Load the trained policy parameters into VLLM
        load_policy_into_vllm_instance(policy_model, llm)
        
        # Sample a subset for evaluation (to speed up and save memory)
        eval_indices = random.sample(range(len(val_prompts)), min(len(val_prompts), 5))  # Reduce to 20 samples
        eval_prompts = [val_prompts[i] for i in eval_indices]
        eval_answers = [val_answers[i] for i in eval_indices]
        
        # Generate responses
        sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
        outputs = llm.generate(eval_prompts, sampling_params)
        
        # Calculate accuracy and show examples
        correct = 0
        print(f"\n=== Epoch {epoch} - Validation Examples ===")
        
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            true_answer = eval_answers[i].strip()
            
            # Extract the final answer (assuming it's in a boxed format)
            answer_match = re.search(r'\\boxed\{([^}]+)\}', generated_text)
            if answer_match:
                predicted_answer = answer_match.group(1).strip()
                is_correct = predicted_answer == true_answer
                if is_correct:
                    correct += 1
            else:
                predicted_answer = "No boxed answer found"
                is_correct = False
            
            # Show first 3 examples
            if i < 3:
                print(f"\n--- Example {i+1} ---")
                print(f"Question: {eval_prompts[i][:200]}...")
                print(f"Generated Response: {generated_text[:300]}...")
                print(f"True Answer: {true_answer}")
                print(f"Predicted Answer: {predicted_answer}")
                print(f"Correct: {is_correct}")
        
        accuracy = correct / len(eval_prompts)
        print(f"\nEpoch {epoch} - Validation Accuracy: {accuracy:.4f} ({correct}/{len(eval_prompts)})")
        print("=" * 50)
        

        
    except ImportError:
        print("VLLM not available, skipping evaluation")
    except Exception as e:
        print(f"Evaluation error: {e}")
    finally:
        # Set model back to training mode after evaluation
        policy_model.train()
        # Ensure CUDA_VISIBLE_DEVICES is restored even if an error occurs
        pass

def sft_experiment():
    # Initialize VLLM for evaluation (only once)
    print("Initializing VLLM for evaluation...")
    from vllm import LLM
    



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
            gpu_memory_utilization=0.7  # Reduce memory usage for vLLM
        )
        print("✓ vLLM inference model initialized successfully")
    except Exception as e:
        print(f"vLLM initialization failed: {e}")
        raise
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-1.5B')
    
    # Load training and validation data
    train_data = load_math_data('data/MATH/train.jsonl')
    val_data = load_math_data('data/MATH/validation.jsonl')
    
    # Prepare training prompts and answers
    train_prompts = []
    train_answers = []
    
    print("Preparing training data and analyzing sequence lengths...")
    sequence_lengths = []
    
    for i, p in enumerate(train_data):
        prompt_string = R1_ZERO_PROMPT.format(
            question=p['problem']
        )
        train_prompts.append(prompt_string)
        train_answers.append(p['answer'])
        
        # 分析序列长度（只分析前1000个样本以节省时间）
        prompt_tokens = tokenizer(prompt_string)['input_ids']
        answer_tokens = tokenizer(p['answer'])['input_ids']
        total_length = len(prompt_tokens) + len(answer_tokens)
        sequence_lengths.append(total_length)
    
    # 打印序列长度统计信息
    if sequence_lengths:
        sequence_lengths.sort()
        print(f"Sequence length statistics (first 1000 samples):")
        print(f"  Min: {min(sequence_lengths)}")
        print(f"  Max: {max(sequence_lengths)}")
        print(f"  Mean: {sum(sequence_lengths)/len(sequence_lengths):.1f}")
        print(f"  Median: {sequence_lengths[len(sequence_lengths)//2]}")
        print(f"  95th percentile: {sequence_lengths[int(len(sequence_lengths)*0.95)]}")
        print(f"  99th percentile: {sequence_lengths[int(len(sequence_lengths)*0.99)]}")
        
        # 建议合适的最大序列长度
        suggested_max_len = min(2048, sequence_lengths[int(len(sequence_lengths)*0.95)])
        print(f"  Suggested max length: {suggested_max_len}")
    else:
        suggested_max_len = 1024
        print(f"Using default max length: {suggested_max_len}")
    
    # Prepare validation prompts and answers
    val_prompts = []
    val_answers = []
    
    for p in val_data:
        prompt_string = R1_ZERO_PROMPT.format(
            question=p['problem']
        )
        val_prompts.append(prompt_string)
        val_answers.append(p['answer'])
    
    # 设置模型为训练模式
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5,eps=1e-6)

    # 设置最大序列长度限制（基于数据分析结果）
    MAX_SEQUENCE_LENGTH = suggested_max_len
    print(f"Using max sequence length: {MAX_SEQUENCE_LENGTH}")
    
    for epoch in range(200000):
        # Clear GPU cache before each epoch
        torch.cuda.empty_cache()
        
        # Randomly sample training prompts and answers (reduce batch size to save memory)
        indices = random.sample(range(len(train_prompts)), min(len(train_prompts), 8))  # 进一步减少到2个样本
        prompt_strs = [train_prompts[i] for i in indices]
        output_strs = [train_answers[i] for i in indices]
        
        model_device = next(model.parameters()).device
        
        # 使用带序列长度限制的tokenize函数
        res = tokenize_prompt_and_output_with_limit(
            prompt_strs, output_strs, tokenizer, 
            device=model_device, max_length=MAX_SEQUENCE_LENGTH
        )
        
        input_ids = res['input_ids']
        labels = res['labels']
        response_mask = res['response_mask']
        
        # 检查显存使用情况
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(model_device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(model_device) / 1024**3   # GB
            print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        
        try:
            policy_log_probs = get_response_log_probs(model, input_ids, labels)['log_probs']
            optimizer.zero_grad()
            loss, _ = sft_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps=1, normalize_constant=1.0)
            optimizer.step()
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
        except torch.cuda.OutOfMemoryError as e:
            print(f"❌ OOM at epoch {epoch}: {e}")
            print("Skipping this batch and reducing sequence length...")
            MAX_SEQUENCE_LENGTH = max(512, MAX_SEQUENCE_LENGTH // 2)  # 动态减少序列长度
            print(f"New max sequence length: {MAX_SEQUENCE_LENGTH}")
            continue
        
        # Clear intermediate variables to free memory
        del input_ids, labels, response_mask, policy_log_probs, loss
        
        if (epoch+1) % 200 == 0:
            # Save model
            model.save_pretrained(f'sft_model')
            tokenizer.save_pretrained(f'sft_model')
            print(f'Model saved to sft_model')
        
        if (epoch+1) % 2 == 0:
            # Clear GPU cache before evaluation
            torch.cuda.empty_cache()
            
        
        if (epoch+1) % 100 == 0:
            # Evaluate on validation set using VLLM
            evaluate_with_vllm(val_prompts, val_answers, epoch, model, vllm_model)
            
            # Clear GPU cache after evaluation
            torch.cuda.empty_cache()

if __name__ == '__main__':
    sft_experiment()