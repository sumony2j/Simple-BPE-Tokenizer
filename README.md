# 🧠 Byte Pair Encoding (BPE) Tokenizer from Scratch

This project implements a Byte Pair Encoding (BPE) tokenizer entirely from scratch using pure Python. It allows you to train on any raw UTF-8 text file and use the trained tokenizer for encoding/decoding strings.

---

## 📘 What is Byte Pair Encoding (BPE)?

Byte Pair Encoding is a data compression technique adapted for NLP tokenization. It begins with a vocabulary of all single characters (bytes `0–255`) and iteratively merges the most frequent **adjacent byte-pairs** to form new tokens. This helps reduce vocabulary size while preserving the ability to recover original text.

BPE is widely used in models like GPT, RoBERTa, and OpenNMT.

---

## 📈 BPE Tokenizer Workflow

### 🔧 Step 1: Initialize
- Load your training text dataset (UTF-8).
- Convert the text to a list of byte IDs (`0–255`).

### 🔍 Step 2: Count Pairs
- Iterate through all sequences in the corpus.
- Count frequency of all adjacent token (byte) pairs.

### 🔄 Step 3: Merge Most Frequent Pair
- Find the most common pair.
- Assign a new token ID (starting from `256`).
- Replace all instances of that pair with the new token.
- Store the merge in `merging_rules`.

### 📚 Step 4: Build Vocabulary
- Start with `{0..255}` as base vocabulary.
- Add merged tokens as they are created.
- Final vocabulary maps token ID → byte sequence.

### 💾 Step 5: Save Tokenizer
The tokenizer is saved as a `.bin` file using Python's `pickle` module. It contains:
```python
{
  "merging_rules": { (108, 108): 256, (256, 111): 257, ... },
  "vocabulary":     { 0: b'\x00', ..., 256: b'll', 257: b'llo', ... }
}
```

---

## 🧪 Encoding & Decoding

### 🔐 Encoding
- Input text is first converted to byte tokens.
- Apply merge rules **in order** to repeatedly join token pairs.
- Result: List of integer token IDs.

### 🔓 Decoding
- Each token ID maps to a byte sequence.
- Join all byte sequences and decode to UTF-8 string.

---

## 🧠 Tokenizer Internals

| Function                | Description                                             |
|-------------------------|---------------------------------------------------------|
| `get_pairs()`           | Counts frequency of all adjacent byte/token pairs       |
| `merge_tokens()`        | Merges selected pair and updates token sequences        |
| `train_tokenizer()`     | Runs BPE loop until `vocab_size` is reached             |
| `build_vocabulary()`    | Maps token ID → bytes                                   |
| `encoder(text)`         | Tokenizes input string to token ID list                 |
| `decoder(token_ids)`    | Converts token IDs back to original string              |
| `save_tokenizer(path)`  | Saves model as `.bin` with pickle                       |
| `load_tokenizer(path)`  | Loads model from `.bin` file                            |

---

## 🖥️ Command-Line Usage

### ✅ Train a Tokenizer

```bash
python Tokenizer.py --train --dataset ./train.txt --vocab_size 300 --save my_tokenizer.bin
```

### ✅ Use the Tokenizer

```bash
python Tokenizer.py --use_tokenizer --load my_tokenizer.bin --input "Hello world"
```

Or to tokenize a text file:

```bash
python Tokenizer.py --use_tokenizer --load my_tokenizer.bin --input ./test.txt
```

---

## 🧾 CLI Arguments

| Argument            | Type      | Required? | Description |
|---------------------|-----------|-----------|-------------|
| `--dataset`         | `str`     | No        | Path to the training dataset file (UTF-8 text). Default: `./train.txt` |
| `--save`            | `str`     | No        | Filepath to save the trained tokenizer model (`.bin`). Default: `./tokenizer_model.bin` |
| `--load`            | `str`     | No        | Filepath to load a previously trained tokenizer model (`.bin`). Default: `./tokenizer_model.bin` |
| `--use_tokenizer`   | `flag`    | No        | Run tokenizer on input (must use `--input` with it) |
| `--vocab_size`      | `int`     | No        | Total desired vocabulary size (minimum 256). Default: `300` |
| `--train`           | `flag`    | No        | Train a new tokenizer on the provided `--dataset` |
| `--input`           | `str`     | Required with `--use_tokenizer` | Raw text string or path to file to be tokenized and decoded |

> Note: The `.bin` file must be a valid pickle file with `merging_rules` and `vocabulary` keys.

---

## 💾 File Format (.bin)

Saved tokenizer file is a Python dictionary serialized with `pickle`:
```python
{
  "merging_rules": { (byte1, byte2): new_token_id, ... },
  "vocabulary":    { token_id: b'some_bytes', ... }
}
```

- `merging_rules` maintains training history of BPE merges.
- `vocabulary` allows reversible decoding.

---

## 🌟 Features

- ✅ Simple BPE logic in Python
- ✅ UTF-8 safe and reversible decoding
- ✅ CLI interface for training and inference
- ✅ Clean token merging logic
- ✅ Save/load models for reuse

---

## 🔮 Future Roadmap

- [ ] Add visual tokenizer merge graphs
- [ ] Export vocab to JSON or text
- [ ] Batch file tokenization
- [ ] Streamlit/Gradio GUI
- [ ] NumPy acceleration
- [ ] Unit test coverage

---

## ⚖️ License

MIT License.  
Use freely with attribution.

---

## 🙏 Acknowledgments

- Inspired by OpenAI’s GPT tokenizers  
- Inspired by and builds upon the insightful work and educational content shared by **Andrej Karpathy**, especially his [YouTube tutorials](https://www.youtube.com/@AndrejKarpathy). His clear explanations and practical examples have been invaluable in understanding and implementing Byte Pair Encoding (BPE) tokenizers.
- CLI polish with `argparse` & `tqdm`  

---
