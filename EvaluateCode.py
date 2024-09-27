#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

# In[5]:


print(transformers.__version__)


# In[6]:


draft_model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
# draft_model_name ="codellama/CodeLlama-7b-hf"
# draft_model_name = "bigcode/starcoderbase-1b"
# draft_model_name = "deepseek-ai/deepseek-coder-1.3b-base"
draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name, trust_remote_code=True, torch_dtype=torch.float16, use_flash_attention_2=True, device_map="auto")#, load_in_4bit=True)
# print(draft_model.device)


# In[ ]:


model_name = "deepseek-ai/deepseek-coder-33b-instruct"
# model_name="codellama/CodeLlama-70b-hf"
# model_name = "bigcode/starcoderbase"
# model_name = "deepseek-ai/deepseek-coder-33b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16,  use_flash_attention_2=True, device_map="auto")#, load_in_4bit=True)#  , use_flash_attention=True)


# In[ ]:


# !wget https://huggingface.co/datasets/m-a-p/CodeEditorBench/resolve/71a18c2fd4896b72b9ef051e75ec2621e2fa5903/code_debug_primary.jsonl


# In[ ]:


from datasets import load_dataset

# dataset_name = "nuprl/CanItEdit"
# dataset_split = "test"
dataset_name = "vdaita/edit_time_5k"
dataset_split = "train"
ds = load_dataset(dataset_name, split=dataset_split)
# ds = load_dataset("m-a-p/CodeEditorBench", split="test").shuffle(seed=42).select(list(range(200)))
# ds = load_dataset("json", data_files="code_debug_primary.jsonl").shuffle(seed=42)["train"].filter(lambda example: len(example["incorrect_solutions"]) > 350).select(list(range(200)))


# In[ ]:


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

        # Create sliding windows of size ngram_size
        windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

        # Convert ngram to a tensor for comparison
        ngram_tensor = torch.tensor(ngram, device=input_ids.device).unsqueeze(0)

        # Find where the windows match the ngram
        matches = (windows == ngram_tensor).all(dim=2)

        # Get the indices of matches
        match_indices = matches.nonzero(as_tuple=True)[1]

        # Iterate through match indices to find a valid continuation
        for idx in match_indices:
            start_idx = idx + ngram_size
            end_idx = start_idx + num_pred_tokens
            # Ensure we don't go beyond the length of input_ids and avoid self-match
            # if end_idx <= input_length and start_idx < input_length - ngram_size:
            #     return input_ids[0, start_idx:end_idx]
            if start_idx < input_length - ngram_size:
                return input_ids[0, start_idx:min(end_idx, input_length)]

    # If no match is found, return an empty tensor
    return torch.tensor([100], dtype=torch.long, device=input_ids.device)

# In[ ]:


from transformers.generation.candidate_generator import CandidateGenerator, _crop_past_key_values
from transformers.generation.stopping_criteria import StoppingCriteria
from transformers.generation.configuration_utils import GenerationConfig
from typing import Tuple, Optional
import time

class DiffPromptLookupCandidateGenerator(CandidateGenerator):
    def __init__(self, input_ids, code_ids, ngram_size=3, num_pred_tokens=10, use_diff=False):
        self.code_ids = code_ids
        self.orig_input_len = input_ids.shape[-1]
        self.ngram_size = ngram_size
        self.num_pred_tokens = num_pred_tokens
        self.last_predicted = 0
        self.use_diff = use_diff
    
    def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        # print("Getting candidates")

        if self.use_diff:
            raise Exception("Diff not implemented")
        else:
            new_tokens = find_candidate_pred_tokens(input_ids, self.ngram_size, self.num_pred_tokens).unsqueeze(0).to(input_ids.device)
        self.last_predicted = new_tokens.shape[-1]
        
        return torch.cat(
            (
                input_ids,
                new_tokens
            ),
            dim=-1
        ), None
    
    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int): # Maybe use the number of matches/scores to have a threshold
        pass
        # if num_matches == self.last_predicted:
        #     self.num_pred_tokens *= 1.5
        # else:
        #     self.num_pred_tokens /= 1.5
        # self.num_pred_tokens = int(self.num_pred_tokens)
        # self.num_pred_tokens = min(self.num_pred_tokens, 100)
        # self.num_pred_tokens = max(self.num_pred_tokens, 1)

class NumRunsStoppingCriteria(StoppingCriteria):
    def __init__(self, max_num_runs=4):
        self.max_num_runs = max_num_runs
        self.num_runs = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        self.num_runs += 1
        return self.num_runs >= self.max_num_runs

def _get_default_candidate_generator_generator(generator: CandidateGenerator):
    def _get_candidate_generator(self, **kwargs):
        return generator
    return _get_candidate_generator

class CodeTwoLayerLookupCandidateGenerator(CandidateGenerator):
    def __init__(self, tokenizer, prompt_tokens, draft_model, input_ids, code_ids, use_score_check=False, min_score=0, scores_count=0, num_runs=4, **diff_prompt_args):
        self.tokenizer = tokenizer
        self.prompt_tokens = prompt_tokens
        self.draft_model = draft_model
        self.input_ids = input_ids
        self.code_ids = code_ids
        self.candidate_generator = DiffPromptLookupCandidateGenerator(
            self.input_ids, 
            self.code_ids,
            **diff_prompt_args
        )
        self.draft_model.generation_config.pad_token_id = tokenizer.pad_token_id
        
        self.past_key_values = None
        self.num_runs = num_runs

        self.draft_model._get_candidate_generator = (_get_default_candidate_generator_generator(self.candidate_generator)).__get__(self.draft_model, type(self.draft_model))

        self.start_token_index = self.input_ids.shape[-1]
        self.min_score = min_score
        self.scores_count = scores_count

        self.use_score_check = use_score_check
    
    def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        if self.past_key_values:
            self.past_key_values = _crop_past_key_values(self.draft_model, self.past_key_values, input_ids.shape[-1] - 1)

        stopping_criteria = [NumRunsStoppingCriteria(self.num_runs), 
                            ]
        if self.use_score_check:
            raise Exception("Score check not implemented")

        # if self.past_key_values:
        #     print(self.past_key_values[0][0].shape)

        old_device = input_ids.device
        input_ids = input_ids.to(self.draft_model.device)

        if self.past_key_values: 
            generation = self.draft_model.generate(
                inputs=input_ids,
                attention_mask=torch.ones(input_ids.shape[-1], device=input_ids.device).unsqueeze(0),
                prompt_lookup_num_tokens=1,
                max_new_tokens=1000,
                stopping_criteria=stopping_criteria,
                past_key_values=self.past_key_values,
                use_cache=True,
                # output_logits=True,
                output_scores=True,
                return_dict_in_generate=True
            )
        else:
            generation = self.draft_model.generate(
                inputs=input_ids,
                attention_mask=torch.ones(input_ids.shape[-1], device=input_ids.device).unsqueeze(0),
                prompt_lookup_num_tokens=1,
                max_new_tokens=1000,
                stopping_criteria=stopping_criteria,
                use_cache=True,
                # output_logits=True,
                output_scores=True,
                return_dict_in_generate=True
            )

        input_ids = input_ids.to(old_device)
        # print("Scores: ", generation.scores)

        self.pred_tokens_count = generation.sequences.shape[-1] - input_ids.shape[-1]
        self.past_key_values = generation.past_key_values
        self.past_top_scores = torch.stack(generation.scores, dim=1).max(dim=1).values[0]

        return generation.sequences, torch.stack(generation.scores, dim=1)

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        pass 


# In[ ]:


def print_update(dictionary):
    for key in dictionary:
        if len(dictionary[key]) > 0:
            print("\t", key, ": ", str(dictionary[key][-1])[:min(50, len(str(dictionary[key][-1])))])
    print("======")


# In[ ]:


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


# In[ ]:

import json
from tqdm import tqdm
from transformers import TextStreamer
from rapidfuzz.distance import Levenshtein
import difflib
from torch.profiler import profile, record_function, ProfilerActivity

# lookup_tokens = [10, 20, 40, 60, 80, 100, 120]
# lookup_tokens = [40]
# lookup_tokens = [40, 80, 120]
# lookup_tokens = [1]

model_draft_tokens = [1, 2, 4, 8, 12, 16]
# model_draft_tokens = [1]
# lookup_tokens = [80]
lookup_tokens = [20, 40, 60, 80, 120, 160, 200]
stats = {mdt: {lt: {"method": [], "method_diff": [], "assisted": [], "pld": [], "regular": [], "lev_similarity": [], "generated_tokens_pld": [], "generated_tokens_method": [], "diff": [], "method_output": [], "regular_output": [], "pld_output": [], "generated_tokens_regular": []} for lt in lookup_tokens} for mdt in model_draft_tokens}

global_min_score = 0
global_scores_count = 0

regular_get_candidate_generator = model._get_candidate_generator

for mdt in model_draft_tokens:
    for lt in lookup_tokens:
        for row_idx, row in tqdm(enumerate(ds)):
            input_text = shot + "\n## Code Before:\n{code_text}\n## Instruction: {question}\n## Code After:\n".format(code_text=row['code'], question=row['change_request'])
            inputs = tokenizer(input_text, return_tensors="pt")
            inputs = tokenizer.apply_chat_template([
                {
                    "role": "user",
                    "content": input_text
                },
            ], tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
            #     response_prompt = tokenizer.encode("""Sure, here is the modified code:

            # ```python
            # """, return_tensors="pt").to(model.device)[:, 1:]
            # inputs = torch.cat((inputs, response_prompt), dim=-1)

            # input_text = f"<commit_before>\n{row['incorrect_solutions']}\n<commit_msg>\nFix error {row['type']}\n<commit_after>\n"
            # input_text = f"## Code Before:\n{row['incorrect_solutions']}\n## Change Requested:\nFix error {row['type']}\n## Code After:\n"
            # inputs = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

            code_tokens = tokenizer(row['code'], return_tensors="pt").to(model.device)
            starting_input_tokens = inputs.shape[-1]

            max_new_tokens = code_tokens.input_ids.shape[-1] + 500

            model._get_candidate_generator = (regular_get_candidate_generator).__get__(model, type(model))

            # Use HuggingFace assisted decoding
            # start_time = time.perf_counter()
            # assisted_output = model.generate(
            #     input_ids=inputs,
            #     max_new_tokens=max_new_tokens,
            #     stopping_criteria=[CodeContentStoppingCriteria(tokenizer, inputs.shape[-1])],
            #     return_dict_in_generate=True,
            #     output_scores=True,
            #     assistant_model=draft_model
            # )
            # end_time = time.perf_counter()
            # stats[lt]["assisted"].append(end_time - start_time)

            # # Use HuggingFace prompt lookup decoding
            if mdt == 1:
                start_time = time.perf_counter()
                pld_output = model.generate(
                    input_ids=inputs,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    prompt_lookup_num_tokens=lt,
                    use_cache=True
                )
                end_time = time.perf_counter()
                stats[mdt][lt]["pld"].append(end_time - start_time)
                stats[mdt][lt]["generated_tokens_pld"].append(pld_output.sequences[:, starting_input_tokens:].shape[-1])
                stats[mdt][lt]["pld_output"].append(tokenizer.batch_decode(pld_output.sequences[:, starting_input_tokens:])[0])

            # # # Use regular HuggingFace text generation
            if mdt == 1 and lt == 20:
                start_time = time.perf_counter()
                regular_outputs = model.generate(
                    input_ids=inputs,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cache=True,
                )
                end_time = time.perf_counter()
                stats[mdt][lt]["regular"].append(end_time - start_time)
                stats[mdt][lt]["regular_output"].append(tokenizer.batch_decode(regular_outputs.sequences[:, starting_input_tokens:])[0])
                stats[mdt][lt]["generated_tokens_regular"].append(regular_outputs.sequences[:, starting_input_tokens].shape[-1])

            # new_text = tokenizer.batch_decode(pld_output.sequences[:, starting_input_tokens:])[0]

            # print(row['before'], new_text)

            # lev_similarity = Levenshtein.normalized_similarity(row['code'], new_text)
            # stats[mdt][lt]["lev_similarity"].append(lev_similarity)

            # stats[lt]["generated_tokens"].append(pld_output.sequences.shape[-1])\

            # # Two Layer Lookup Candidate Generator without Score Check
            if lt == 80:
                two_layer_candidate_generator = CodeTwoLayerLookupCandidateGenerator(
                    tokenizer,
                    inputs.shape[-1],
                    draft_model,
                    inputs,
                    code_tokens.input_ids.tolist()[0],
                    use_diff=False,
                    use_score_check=False,
                    min_score=global_min_score,
                    scores_count=global_scores_count,
                    ngram_size=5,
                    num_pred_tokens=lt,
                    num_runs=mdt,
                )
                model._get_candidate_generator = (_get_default_candidate_generator_generator(two_layer_candidate_generator)).__get__(model, type(model))

                global_min_score = two_layer_candidate_generator.min_score
                global_scores_count = two_layer_candidate_generator.scores_count
                start_time = time.perf_counter()
                # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                test_out = model.generate(
                    inputs=inputs,
                    prompt_lookup_num_tokens=1,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    # streamer=TextStreamer(tokenizer)
                )
                # torch.cuda.synchronize()
                end_time = time.perf_counter()
                # prof.export_chrome_trace(f"profiler_results/method_draft_runs_{mdt}_row_{row_idx}.json")
                stats[mdt][lt]["method"].append(end_time - start_time)
                two_layer_result = tokenizer.batch_decode(test_out[:, starting_input_tokens:])[0]
                stats[mdt][lt]["generated_tokens_method"].append(test_out[:, starting_input_tokens:].shape[-1])
                stats[mdt][lt]["method_output"].append(two_layer_result)

            # if not(new_text.strip() == two_layer_result.strip()):
            # print("Results with differences:")
            # stats[mdt][lt]["diff"].append("\n".join(difflib.unified_diff(new_text.splitlines(), two_layer_result.splitlines(), n=3)))
            # print("=======================")

            # Method with diff
            # two_layer_candidate_generator = CodeTwoLayerLookupCandidateGenerator(
            #     tokenizer,
            #     inputs.shape[-1],
            #     draft_model,
            #     inputs,
            #     code_tokens.input_ids.tolist()[0],
            #     use_diff=True,
            #     use_score_check=False,
            #     min_score=global_min_score,
            #     scores_count=global_scores_count,
            #     ngram_size=5,
            #     num_pred_tokens=lt
            # )
            # model._get_candidate_generator = (_get_default_candidate_generator_generator(two_layer_candidate_generator)).__get__(model, type(model))

            # global_min_score = two_layer_candidate_generator.min_score
            # global_scores_count = two_layer_candidate_generator.scores_count
            # start_time = time.perf_counter()
            # test_out = model.generate(
            #     inputs=inputs,
            #     prompt_lookup_num_tokens=1,
            #     max_new_tokens=max_new_tokens,
            #     stopping_criteria=[CodeContentStoppingCriteria(tokenizer, inputs.shape[-1])],
            #     use_cache=True,
            #     # streamer=TextStreamer(tokenizer)
            # )
            # end_time = time.perf_counter()
            # stats[lt]["method_diff"].append(end_time - start_time)

            print_update(stats[mdt][lt])
            temp_save_file = open(f"temp_save_stats_joint_pld_method.json", "w+")
            temp_save_file.write(json.dumps(stats))
            temp_save_file.close()

print(stats)


# In[ ]:


model_name_underscore = model_name.replace("/", "_")
dataset_name_underscore = dataset_name.replace("/", "_")

stats_file = open(f"stats_{model_name_underscore}_{dataset_name_underscore}_joint_pld_method.json", "w+")
stats_file.write(json.dumps(stats))
stats_file.close()


# In[ ]:
