import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers 
from transformers.generation.candidate_generator import (
    CandidateGenerator,
    _crop_past_key_values,
)
from transformers.generation.stopping_criteria import StoppingCriteria
from transformers.generation.configuration_utils import GenerationConfig
from typing import Tuple, Optional
import time
from typing import List
from difflib import SequenceMatcher
load_file = "temp_save_stats_joint_pld_method.json"

draft_model_name = "deepseek-ai/deepseek-coder-33b-instruct"
model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name, trust_remote_code=True, torch_dtype=torch.float16, use_flash_attention_2=True, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    use_flash_attention_2=True,
    device_map="auto",
)  #

from datasets import load_dataset
dataset_name = "vdaita/edit_time_5k"
dataset_split = "train"
ds = load_dataset(dataset_name, split=dataset_split)

import difflib
@torch.no_grad()
def find_candidate_pred_tokens(input_ids, max_ngram_size=3, num_pred_tokens=10):
    input_length = input_ids.size(1)
    # Ensure max_ngram_size and num_pred_tokens are valid
    if max_ngram_size <= 0 or num_pred_tokens <= 0 or max_ngram_size > input_length:
        raise ValueError("Invalid max_ngram_size or num_pred_tokens")
    for ngram_size in range(max_ngram_size, 0, -1):
        # Extract the last n tokens as our search ngram
        ngram = input_ids[0, -ngram_size:].tolist()
        windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)
        ngram_tensor = torch.tensor(ngram, device=input_ids.device).unsqueeze(0)
        matches = (windows == ngram_tensor).all(dim=2)
        match_indices = matches.nonzero(as_tuple=True)[1]
        for idx in match_indices:
            start_idx = idx + ngram_size
            end_idx = start_idx + num_pred_tokens
            if start_idx < input_length - ngram_size:
                return input_ids[0, start_idx : min(end_idx, input_length)]
    # If no match is found, return an empty tensor
    return torch.tensor([100], dtype=torch.long, device=input_ids.device)

class CodeContentStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_tokens: int, newline_count=5):
        self.newline_token = tokenizer.encode("""
""")[-1]
        self.code_block_token = tokenizer.encode("```")[-1]        
        self.newline_count = newline_count
        self.prompt_tokens = prompt_tokens
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        considered_tokens = input_ids[:, self.prompt_tokens:][0]
        return (self.code_block_token == considered_tokens).any().item()

class NumRunsStoppingCriteria(StoppingCriteria):
    def __init__(self, max_num_runs=4):
        self.max_num_runs = max_num_runs
        self.num_runs = 0

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> torch.BoolTensor:
        self.num_runs += 1
        return self.num_runs >= self.max_num_runs


class DiffPromptLookupCandidateGenerator(CandidateGenerator):
    def __init__(
        self, input_ids, code_ids, ngram_size=3, num_pred_tokens=10, use_diff=False
    ):
        self.code_ids = code_ids
        self.orig_input_len = input_ids.shape[-1]
        self.ngram_size = ngram_size
        self.num_pred_tokens = num_pred_tokens
        self.last_predicted = 0
        self.use_diff = use_diff

    def get_candidates(
        self, input_ids: torch.LongTensor
    ) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        # print("Getting candidates")
        new_tokens = find_candidate_pred_tokens(
            input_ids, self.ngram_size, self.num_pred_tokens
        ).unsqueeze(0)
        self.last_predicted = new_tokens.shape[-1]

        return torch.cat((input_ids, new_tokens), dim=-1), None

    def update_candidate_strategy(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int
    ):  # Maybe use the number of matches/scores to have a threshold
        pass


def _get_default_candidate_generator_generator(generator: CandidateGenerator):
    def _get_candidate_generator(self, **kwargs):
        return generator
    return _get_candidate_generator

class PrecisePromptLookup(CandidateGenerator):
    def __init__(
        self,
        tokenizer,
        prompt_tokens,
        copy_ranges: List[int],
        draft_model,
        input_ids,
        code_ids,
        min_score=0,
        scores_count=0,
        num_runs=4,
        **diff_prompt_args
    ):
        self.tokenizer = tokenizer
        self.prompt_tokens = prompt_tokens
        self.copy_ranges = copy_ranges # (start_token, distance of copy) until the next start token is reached
        self.draft_model = draft_model
        self.input_ids = input_ids
        self.code_ids = code_ids
        self.candidate_generator = DiffPromptLookupCandidateGenerator(
            self.input_ids, self.code_ids, **diff_prompt_args
        )
        self.draft_model.generation_config.pad_token_id = tokenizer.pad_token_id

        self.past_key_values = None
        self.num_runs = num_runs

        self.draft_model._get_candidate_generator = (
            _get_default_candidate_generator_generator(self.candidate_generator)
        ).__get__(self.draft_model, type(self.draft_model))

        self.start_token_index = self.input_ids.shape[-1]
        self.min_score = min_score
        self.scores_count = scores_count

    def get_candidates(
        self, input_ids: torch.LongTensor
    ) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        if self.past_key_values:
            self.past_key_values = _crop_past_key_values(
                self.draft_model, self.past_key_values, input_ids.shape[-1] - 1
            )

        # Number of generated tokens - check if this is something that has already been accounted for
        generated_tokens_so_far = input_ids.shape[-1] - self.start_token_index
        for chunk in self.copy_ranges:
            if chunk[0] + 5 <= generated_tokens_so_far and chunk[0] + chunk[1] < generated_tokens_so_far: # make sure that the starting bit gets generated
                # copy over a large chunk and that's it
                new_tokens = find_candidate_pred_tokens(
                    input_ids, 5, chunk[1]
                ).unsqueeze(0)
                if new_tokens.shape[-1] == 1: # if there wasn't a match, pause
                    continue
                return torch.cat((input_ids, new_tokens), dim=-1), None

        stopping_criteria = [
            NumRunsStoppingCriteria(self.num_runs),
        ]
        
        old_device = input_ids.device
        input_ids = input_ids.to(self.draft_model.device)
        if self.past_key_values:
            generation = self.draft_model.generate(
                inputs=input_ids,
                attention_mask=torch.ones(
                    input_ids.shape[-1], device=input_ids.device
                ).unsqueeze(0),
                prompt_lookup_num_tokens=1,
                max_new_tokens=1000,
                stopping_criteria=stopping_criteria,
                past_key_values=self.past_key_values,
                use_cache=True,
                # output_logits=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        else:
            generation = self.draft_model.generate(
                inputs=input_ids,
                attention_mask=torch.ones(
                    input_ids.shape[-1], device=input_ids.device
                ).unsqueeze(0),
                prompt_lookup_num_tokens=1,
                max_new_tokens=1000,
                stopping_criteria=stopping_criteria,
                use_cache=True,
                # output_logits=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        input_ids = input_ids.to(old_device)

        self.pred_tokens_count = generation.sequences.shape[-1] - input_ids.shape[-1]
        self.past_key_values = generation.past_key_values
        self.past_top_scores = (
            torch.stack(generation.scores, dim=1).max(dim=1).values[0]
        )

        return generation.sequences, torch.stack(generation.scores, dim=1)

    def update_candidate_strategy(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int
    ):
        pass

shot = """## Code Before:
def add(a, b):
    return a + b
## Instruction:
Add a "sub" function that subtracts two numbers. Also write docstrings for both functions and change a,b to x,y.
## Code After:
def add(x, y):
    \"\"\"Adds two numbers.\"\"\"
    return x + y

def sub(x, y):
    \"\"\"Subtracts two numbers.\"\"\"
    return x - y"""


from tqdm import tqdm
import json

regular_outputs = json.loads(open("temp_save_stats_joint_pld_method.json", "r").read())["1"]["20"]["regular_output"]

results = []
save_file = open("optimal_breaks_deepseek-ai_deepseek-coder-33b-instruct_vdaita_edit_time_5k.json", "w+")

for (row, original_output) in tqdm(zip(ds, regular_outputs)):
    input_text = shot + "\n## Code Before:\n{code_text}\n## Instruction: {question}\n## Code After:\n".format(code_text=row['code'], question=row['change_request'])
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = tokenizer.apply_chat_template([
        {
            "role": "user",
            "content": input_text
        },
    ], tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

    code_tokens = tokenizer(row['code'], return_tensors="pt")
    starting_input_tokens = inputs.shape[-1]

    max_new_tokens = code_tokens.input_ids.shape[-1] + 500

    tokenized_original_output = tokenizer.encode(original_output)
    code_tokens_list_encoded = tokenizer.encode(row['code'])

    # print(tokenized_origi nal_output)

    copy_ranges = []
    sm = difflib.SequenceMatcher(None, code_tokens_list_encoded, tokenized_original_output) # we made sure in the previous code to have the starting chunk removed when calculating the new output
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            copy_ranges.append((j1, j2 - j1)) # this shows the corresponding range in the generated text, which should line up pretty well

    two_layer_candidate_generator = PrecisePromptLookup(
        tokenizer,
        inputs.shape[-1],
        copy_ranges,
        draft_model,
        inputs,
        code_tokens.input_ids.tolist()[0],
        use_diff=False,
        min_score=0,
        scores_count=0,
        ngram_size=5,
        num_pred_tokens=10 # setting a very small range so that finer-grained changes can be made
    )
    model._get_candidate_generator = (_get_default_candidate_generator_generator(two_layer_candidate_generator)).__get__(model, type(model))

    print(model._get_candidate_generator())

    start_time = time.perf_counter()
    inputs = inputs.to(model.device)
    test_out = model.generate(
        inputs=inputs,
        prompt_lookup_num_tokens=1,
        max_new_tokens=max_new_tokens,
        stopping_criteria=[CodeContentStoppingCriteria(tokenizer, inputs.shape[-1])],
        use_cache=True,
    )
    end_time = time.perf_counter()  # time.time()

    results.append(
        {
            # "input": input_text,
            "output": tokenizer.batch_decode(test_out[:, starting_input_tokens:])[0],
            "time": end_time - start_time,
            # "original_output": original_output,
        }
    )

    save_file.write(json.dumps(results))

# save_file.write(json.dumps(results))
