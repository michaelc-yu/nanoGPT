"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np


input_file_path = './sudoku_vocab.txt'

with open(input_file_path, 'r') as f:
    lines = f.readlines()

# Extract the actual token part
vocab = []
for line in lines:
    parts = line.split("'")  # Split by single quote
    if len(parts) > 1:  # Ensure there is a second part
        vocab.append(parts[1])  # token is the second part

# Print the extracted token values
print(vocab)


# get all the unique characters that occur in this text
vocab_size = len(vocab)
print("all the unique characters:", ''.join(vocab))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(vocab) }
itos = { i:ch for i,ch in enumerate(vocab) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string



# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
print(f"meta: {meta}")
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

