import difflib
from transformers import AutoTokenizer

original_code = """
import numpy as np
import matplotlib.pyplot as plt

# Calculate the average
average_throughput = np.mean(tokens_per_sec_arr)
print(f"Average Throughput: {average_throughput} tokens/sec")

# Plotting the histogram
plt.hist(tokens_per_sec_arr, bins=20, color='blue', edgecolor='black', alpha=0.7)
plt.title('Histogram of Throughput Values')
plt.xlabel('Tokens per Second')
plt.ylabel('Frequency')
plt.axvline(average_throughput, color='red', linestyle='dashed', linewidth=1)
plt.text(average_throughput*0.9, max(plt.ylim())*0.9, f'Average: {average_throughput:.2f}', color = 'red')
plt.show()
"""

generated_text = """
import numpy as np
import matplotlib.pyplot as plt

# Calculate the average
average_throughput = np.mean(tokens_per_sec_arr)
print(f"Average Throughput: {average_throughput} tokens/sec")

# Plotting the histogram
plt.hist(tokens_per_sec_arr, bins=20, color='blue', edgecolor='black', alpha=0.7)
plt.title('Histogram of Throughput Values')
plt.xlabel('Tokens per Second')
plt.ylabel('Frequency')
plt.axvline(average_throughput, color='red', linestyle='dashed', linewidth=1)
plt.text(average_throughput*0.9, max(plt.ylim())*0.9, f'Average: {average_throughput:.2f}', color = 'red')
plt.xlim(0, max(tokens_per_sec_arr))
plt.show()
"""

model_name = "deepseek-ai/deepseek-coder-6.7b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

original_code_tokens = tokenizer.encode(original_code)[1:] # remove the beginning of string token
generated_tokens = tokenizer.encode(generated_text)

# print(original_code_tokens)
# print(generated_tokens)

sm = difflib.SequenceMatcher(None, original_code_tokens, generated_tokens)
# for tag, i1, i2, j1, j2 in sm.get_opcodes():
#     print('{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}'.format(
#         tag, i1, i2, j1, j2, original_code_tokens[i1:i2], generated_tokens[j1:j2]))

deleted = added = changed = same = last_deleted = 0
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == 'replace':
        print("Replace: ", tokenizer.decode(original_code_tokens[i1:i2]), " with: ", tokenizer.decode(original_code_tokens[j1:j2]))
        changed += i2 - i1
    elif tag == 'delete':
        print("Delete: ", tokenizer.decode(original_code_tokens[i1:i2]))
        deleted += i2 - i1
        last_deleted = i2 - i1
    elif tag == 'insert':
        print("Insert: ", tokenizer.decode(generated_tokens[j1:j2]))
        added += j2 - j1
    elif tag == 'equal':
        print("Equal: ", tokenizer.decode(generated_tokens[j1:j2]))
        same += i2 - i1

print(deleted + )