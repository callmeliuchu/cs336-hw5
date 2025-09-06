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
from rl import tokenize_prompt_and_output,get_response_log_probs,sft_microbatch_train_step,load_policy_into_vllm_instance


R1_ZERO_PROMPT = open('cs336_alignment/prompts/r1_zero.prompt', 'r').read()

def load_math_data(data_path):
    """Load MATH dataset from jsonl file"""
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def evaluate_with_vllm(val_prompts, val_answers, epoch, policy_model):
    """Evaluate model using VLLM on CUDA 1"""
    try:
        from vllm import LLM, SamplingParams
        import re
        
        # Set CUDA_VISIBLE_DEVICES to use GPU 1
        original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        
        # Initialize VLLM on CUDA 1 with base model
        llm = LLM(
            model="Qwen/Qwen2.5-Math-1.5B",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8
        )
        
        # Load the trained policy parameters into VLLM
        load_policy_into_vllm_instance(policy_model, llm)
        
        # Sample a subset for evaluation (to speed up)
        eval_indices = random.sample(range(len(val_prompts)), min(len(val_prompts), 100))
        eval_prompts = [val_prompts[i] for i in eval_indices]
        eval_answers = [val_answers[i] for i in eval_indices]
        
        # Generate responses
        sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
        outputs = llm.generate(eval_prompts, sampling_params)
        
        # Calculate accuracy
        correct = 0
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            # Extract the final answer (assuming it's in a boxed format)
            answer_match = re.search(r'\\boxed\{([^}]+)\}', generated_text)
            if answer_match:
                predicted_answer = answer_match.group(1).strip()
                if predicted_answer == eval_answers[i].strip():
                    correct += 1
        
        accuracy = correct / len(eval_prompts)
        print(f'Epoch {epoch} - Validation Accuracy: {accuracy:.4f} ({correct}/{len(eval_prompts)})')
        
        # Restore original CUDA_VISIBLE_DEVICES
        if original_cuda_visible is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        
    except ImportError:
        print("VLLM not available, skipping evaluation")
    except Exception as e:
        print(f"Evaluation error: {e}")
    finally:
        # Ensure CUDA_VISIBLE_DEVICES is restored even if an error occurs
        if 'original_cuda_visible' in locals():
            if original_cuda_visible is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
            else:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)

def sft_experiment():
    # Load training and validation data
    train_data = load_math_data('data/MATH/train.jsonl')
    val_data = load_math_data('data/MATH/validation.jsonl')
    
    # Prepare training prompts and answers
    train_prompts = []
    train_answers = []
    
    for p in train_data:
        prompt_string = R1_ZERO_PROMPT.format(
            question=p['problem']
        )
        train_prompts.append(prompt_string)
        train_answers.append(p['answer'])
    
    # Prepare validation prompts and answers
    val_prompts = []
    val_answers = []
    
    for p in val_data:
        prompt_string = R1_ZERO_PROMPT.format(
            question=p['problem']
        )
        val_prompts.append(prompt_string)
        val_answers.append(p['answer'])

    # Load model and tokenizer on CUDA 0
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Math-1.5B')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-1.5B')
    
    # Move model to CUDA 0
    model = model.to('cuda:0')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(1000):
        # Randomly sample training prompts and answers
        indices = random.sample(range(len(train_prompts)), min(len(train_prompts), 16))
        prompt_strs = [train_prompts[i] for i in indices]
        output_strs = [train_answers[i] for i in indices]
        
        res = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
        policy_log_probs = get_response_log_probs(model, res['input_ids'], res['labels'])['log_probs']
        response_mask = res['response_mask']
        optimizer.zero_grad()
        loss, _ = sft_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps=1, normalize_constant=1.0)
        optimizer.step()
        print(f'Epoch {epoch}, Loss: {loss.item()}')
        
        if (epoch+1) % 2 == 0:
            # Save model
            model.save_pretrained(f'sft_model')
            tokenizer.save_pretrained(f'sft_model')
            print(f'Model saved to sft_model')
            
            # Evaluate on validation set using VLLM
            evaluate_with_vllm(val_prompts, val_answers, epoch, model)

if __name__ == '__main__':
    sft_experiment()