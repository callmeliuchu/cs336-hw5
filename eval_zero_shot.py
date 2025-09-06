
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

from vllm import LLM, SamplingParams
import json
from collections.abc import Callable
from typing import List
import pandas as pd

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

R1_ZERO_PROMPT = open('cs336_alignment/prompts/r1_zero.prompt', 'r').read()


def evaluate_vllm(  
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    sampling_params: SamplingParams
) -> None:
    outputs = vllm_model.generate(prompts, sampling_params)
    results = []
    for output, answer in zip(outputs, answers):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        reward = reward_fn(generated_text, answer)
        results.append({
            'prompt': prompt,
            'response': generated_text,
            'correct_answer': answer,
            'reward': reward,
        })
    return results


def eval_zero_shot():
    with open('data/MATH/validation.jsonl', 'r') as f:
        prompt_data = [json.loads(json_line) for json_line in f]
    
    prompts = []
    answers = []
    
    for p in prompt_data:
        prompt_string = R1_ZERO_PROMPT.format(
            question=p['problem']
        )
        prompts.append(prompt_string)
        answers.append(p['answer'])
    
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024
    )
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True

    llm = LLM(model='Qwen/Qwen2.5-Math-1.5B')

    results = evaluate_vllm(llm, r1_zero_reward_fn, prompts, answers, sampling_params)

    with open('qwen_baseline_perf.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__': 
    eval_zero_shot()