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
from transformers import AutoModelForCausalLM, AutoTokenizer
from rl import tokenize_prompt_and_output,sample_data,get_response_log_probs,sft_microbatch_train_step


R1_ZERO_PROMPT = open('cs336_alignment/prompts/r1_zero.prompt', 'r').read()

def sft_experiment():
    with open('data/MATH/validation.jsonl', 'r') as f:
        prompt_data = [json.loads(json_line) for json_line in f]
    
    # Prepare data in the format expected by sample_data
    data_arr = []
    for p in prompt_data:
        prompt_string = R1_ZERO_PROMPT.format(
            question=p['problem']
        )
        data_arr.append({
            'question': prompt_string,
            'answer': p['answer']
        })
    

    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Math-1.5B')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-1.5B')

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(1000):
        prompt_strs, output_strs = sample_data(data_arr)
        res = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
        policy_log_probs = get_response_log_probs(model, res['input_ids'], res['labels'])['log_probs']
        response_mask = res['response_mask']
        optimizer.zero_grad()
        loss, _ = sft_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps=1, normalize_constant=1.0)
        optimizer.step()
        print(f'Epoch {epoch}, Loss: {loss.item()}')
        if (epoch+1) % 10 == 0:
            ### save model
            model.save_pretrained(f'sft_model')
            tokenizer.save_pretrained(f'sft_model')
            print(f'Model saved to {f'sft_model'}')

if __name__ == '__main__':
    sft_experiment()