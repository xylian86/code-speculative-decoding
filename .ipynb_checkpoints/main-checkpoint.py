from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.generation.utils import _crop_past_key_values
import difflib
import torch.nn.functional as F
import torch

# Borrows from: https://github.com/romsto/Speculative-Decoding

@torch.no_grad
def regular_generate(target_model, tokenizer, input_str, max_seq_len=500):    
    token_ids = tokenizer(input_str, return_tensors="pt").input_ids
    print(token_ids)

    prediction = target_model.generate(input_ids=token_ids, return_dict=True, max_new_tokens=1, return_dict_in_generate=True)
    print("preloop")
    kv_cache = prediction.past_key_values
    token_ids = prediction.sequences
    # kv_cache = target_model
    
    while len(token_ids) < max_seq_len:
        print("loop")
        prediction = target_model.forward(input_ids=token_ids, past_key_values=kv_cache, use_cache=True)
        print(prediction)
        kv_cache = prediction.past_key_values
        token_ids = prediction.sequences
        print(tokenizer.batch_decode(token_ids))
        if token_ids[0, -1] == tokenizer.eos_token or token_ids[0, -1] == tokenizer.pad_token:
            break

    print(tokenizer.batch_decode(token_ids))

def speculative_generate(target_model, tokenizer, input_str, code_str, max_seq_len=500, lookback=10):

    code_ids = tokenizer(code_str, return_tensors="pt").input_ids
    print(code_ids)
    token_ids = tokenizer(input_str, return_tensors="pt").input_ids
    print(token_ids)

    prediction = target_model.generate(input_ids=token_ids, return_dict=True, max_new_tokens=1, return_dict_in_generate=True)
    print("preloop")
    kv_cache = prediction.past_key_values
    token_ids = prediction.sequences
    # kv_cache = target_model
    
    while len(token_ids) < max_seq_len:
        
        guessed_tokens = 0
        if current_position > input_len:
            sm = difflib.SequenceMatcher(None, code_ids, input_ids[0].tolist()[input_len:current_position])

            deleted = added = changed = same = last_deleted = 0
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == 'replace':
                    changed += i2 - i1
                elif tag == 'delete':
                    deleted += i2 - i1
                    last_deleted = i2 - i1
                elif tag == 'insert':
                    added += j2 - j1
                elif tag == 'equal':
                    same += i2 - i1
            
            approx_tokens_original = changed + deleted + same - last_deleted

            max_matches = 0
            best_match_index = -1
            generated_chunk = input_ids[0, max(current_position - lookback, 0):current_position].tolist()
            for code_start_index in range(num_tokens_original, len(code_ids)):
                code_chunk = code_ids[code_start_index: max(code_start_index + lookback, len(code_tokens))]
                matching_tokens = i = 0
                while i < len(generated_chunk) and i < len(code_chunk):
                    if generated_chunk[i] == code_chunk[i]:
                        matching_tokens += 1
                    i += 1
                if matching_tokens > max_matches:
                    max_matches = matching_tokens
                    best_match_index = code_start_index + len(code_chunk)
        
            guessed_tokens = min(max_seq_len - current_position, len(code_tokens) - best_match_index)
            token_ids[0].extend(code_ids[best_match_index:])

        prediction = target_model.forward(
            input_ids=input_ids,

        )
        kv_cache = prediction.past_key_values
        token_ids = prediction.sequences
        print(tokenizer.batch_decode(token_ids))
        if token_ids[0, -1] == tokenizer.eos_token or token_ids[0, -1] == tokenizer.pad_token:
            break

    print(tokenizer.batch_decode(token_ids))


    input_ids = tokenizer(input_str, return_tensors="pt").input_ids
    input_len = input_ids.shape[1]
    code_ids = tokenizer(code_str, return_tensors="pt").input_ids
    code_ids = code_ids[0].tolist()

    target_cache = DynamicCache()
    
    tokens = torch.zeros((1, max_seq_len), device=target_model.device)
    tokens[0, :input_len] = input_ids[0]

    current_position = input_len

    while current_position < max_seq_len:
        # First, try to align the most recent few tokens to the code string
        guessed_tokens = 0
        if current_position > input_len:
            sm = difflib.SequenceMatcher(None, code_ids, input_ids[0].tolist()[input_len:current_position])

            deleted = added = changed = same = last_deleted = 0
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == 'replace':
                    changed += i2 - i1
                elif tag == 'delete':
                    deleted += i2 - i1
                    last_deleted = i2 - i1
                elif tag == 'insert':
                    added += j2 - j1
                elif tag == 'equal':
                    same += i2 - i1
            
            approx_tokens_original = changed + deleted + same - last_deleted

            max_matches = 0
            best_match_index = -1
            
            generated_chunk = input_ids[0, max(current_position - lookback, 0):current_position].tolist()
            for code_start_index in range(num_tokens_original, len(code_ids)):
                code_chunk = code_ids[code_start_index: max(code_start_index + lookback, len(code_tokens))]
                matching_tokens = i = 0
                while i < len(generated_chunk) and i < len(code_chunk):
                    if generated_chunk[i] == code_chunk[i]:
                        matching_tokens += 1
                    i += 1
                if matching_tokens > max_matches:
                    max_matches = matching_tokens
                    best_match_index = code_start_index + len(code_chunk)
        
            guessed_tokens = min(max_seq_len - current_position, len(code_tokens) - best_match_index)
            tokens[:, current_position:current_position + guessed_tokens] = code_tokens[best_match_index:best_match_index + guessed_tokens]

        prediction = target_model(
            input_ids=input_ids[:, :current_position],
            past_key_values=target_cache,
            use_cache=True
        )
        prediction = F.softmax(prediction, dim=-1)

        target_cache = prediction.past_key_values

        max_accepted = current_position
        for i in range(current_position, current_position + guessed_tokens + 1):
            # check the corresponding value of the prediction
            # if it's accepted, push max_accepted by one
            model_token = torch.argmax(prediction[:, i, :], dim=-1).unsqueeze(-1)
            print("Model's predicted token: ", model_token)
            if model_token == input_ids[:, i]:
                max_accepted = i
            else:
                input_ids[:, current_position] = model_token
                input_ids[:, current_position + 1:] = 0
                max_accepted = i # I don't think non-guessed tokens are in the KV cache
                break
        
        target_cache = _crop_past_key_values(target_model, )
    
        current_position = max_accepted + 1
        
        if current_position > max_seq_len or tokens[0, current_position] in [tokenizer.eos_token, tokenizer.pad_token]:
            break
    
    return tokens[0, :current_position]

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, device_map="auto")
input_text = "#write a quick sort algorithm"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
regular_generate(model, tokenizer, input_text)


# inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(target_model.device)
# # tokenizer.eos_token_id is the id of <|EOT|> token
# outputs = target_model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
# print(tokenizer.batch_decode(outputs))
