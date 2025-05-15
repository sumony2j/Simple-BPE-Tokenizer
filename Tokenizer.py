 
from tqdm import tqdm
import argparse
import pickle
import os

class MYBPE():
    def __init__(self,vocab_size,dataset=None):
        self.vocab_size = vocab_size
        if dataset is not None:
            self.dataset = list(dataset.encode("utf-8"))
    def get_pairs(self,dataset):
        """
        Generate and count adjacent token pairs in the token list.
        """
        byte_tokens = list(dataset)
        pairs = {}
        for i in tqdm(range(len(byte_tokens)-1)):
            if (byte_tokens[i],byte_tokens[i+1]) in pairs:
                pairs[(byte_tokens[i],byte_tokens[i+1])] += 1
            else:
                pairs[(byte_tokens[i],byte_tokens[i+1])] = 1
        return pairs
    def merge_tokens(self,tokens,pair,id):
        """
        Replace all occurrences of a token pair with a new token id.
        """
        merged_tokens = []
        i = 0
        while i < len(tokens):
            if (i < len(tokens)-1) and (tokens[i] == pair[0]) and (tokens[i+1] == pair[1]):
                merged_tokens.append(id)
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1
        return merged_tokens
    def train_tokenizer(self):
        """
        Train the BPE tokenizer by iteratively merging frequent token pairs.
        """
        print("\n------ Training BPE Tokenizer ------\n")
        num_merged_tokens = self.vocab_size - 256
        tokens = self.dataset
        self.merging_rules = {}
        for i in tqdm(range(num_merged_tokens)):
            pair_details = self.get_pairs(tokens)
            top_pair = max(pair_details,key=pair_details.get)
            print(f"\nMerging {top_pair} with {i+256}\n")
            tokens = self.merge_tokens(tokens,top_pair,i+256)
            self.merging_rules[top_pair] = i+256
        return self.merging_rules
    def build_vocabulary(self):
        """
        Build the vocabulary mapping from token ID to actual byte sequence.
        """
        print("\n------ Building Vocabulary ------\n")
        self.voc = {i:bytes([i]) for i in tqdm(range(256))}
        for pair,val in tqdm(self.merging_rules.items()):
            self.voc[val] = self.voc[pair[0]] + self.voc[pair[1]]
    def save_tokenizer(self,path):
        with open(path,"wb") as f:
            pickle.dump(
                {
                    "merging_rules" : self.merging_rules,
                    "vocabulary" : self.voc
                }, f)
        print(f"Tokenizer model is saved {path}")
    def load_tokenizer(self,path):
        with open(path,"rb") as f:
            data = pickle.load(f)
            self.merging_rules = data["merging_rules"]
            self.voc = data["vocabulary"]
    def decoder(self,ids):
        """
        Decode a list of token IDs into a UTF-8 string using the vocabulary.
        """
        print("\n------ Decoding ------\n")
        text = b"".join(self.voc[i] for i in ids)
        text = text.decode("utf-8",errors="replace")
        self.text = text
        return self.text
    def encoder(self,text):
        """
        Encode raw UTF-8 text into token IDs using trained merges.
        """
        print("\n------ Encoding The Input ------\n")
        byte_tokens = list(text.encode("utf-8"))
        pbar = tqdm(desc="Encoding...",unit= " ")
        while len(byte_tokens) > 1:
            pairs = self.get_pairs(byte_tokens)
            replace_pair = min(pairs, key=lambda p: self.merging_rules.get(p,float('inf')))
            if replace_pair not in  self.merging_rules:
                break
            byte_tokens = self.merge_tokens(byte_tokens,replace_pair,self.merging_rules[replace_pair])
            pbar.update(1)
        pbar.close()
        self.tokens = byte_tokens
        return self.tokens    

def valid_tokenizer_model(name:str):
    
    ## Check if the file is of .bin type
    
    if not name.endswith(".bin"):
        raise argparse.ArgumentTypeError("File must have a '.bin' extension.")
    
    ## Check the the file exists & if it has the proper required data-structure/format
    
    if os.path.exists(name):
        try:
            with open(name,"rb") as f:
                data = pickle.load(f)
            if not isinstance(data, dict) or "merging_rules" not in data or "vocabulary" not in data:
                raise argparse.ArgumentTypeError("The .bin file must be a pickle containing 'merging_rules' and 'vocabulary'.")
        except Exception as e:
            raise argparse.ArgumentTypeError(f"Invalid .bin file content: {e}") 
    return name
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Byte Pair Encoding Tokenizer")
    parser.add_argument("--dataset", type=str, default="./train.txt",
                        help="Dataset on which tokenizer will be trained (text file)")
    parser.add_argument("--save",default="./tokenizer_model.bin",type=valid_tokenizer_model,
                        help="Save the tokenizer")
    parser.add_argument("--load",default="./tokenizer_model.bin",type=valid_tokenizer_model,
                        help="load the tokenizer")
    parser.add_argument("--use_tokenizer", action="store_true",
                        help="Run the tokenizer on input")
    parser.add_argument("--vocab_size", default=300, type=int,
                        help="Desired vocabulary size (>= 256)")
    parser.add_argument("--train", action="store_true",
                        help="Flag to train a new tokenizer")
    parser.add_argument("--input", type=str,
                        help="File path or raw input string to tokenize")
    args = parser.parse_args()
    
    if args.train:
        with open(args.dataset,"r",encoding="utf-8") as f:
            data = f.read()
        Tokenizer = MYBPE(args.vocab_size,data)
        Tokenizer.train_tokenizer()
        Tokenizer.build_vocabulary()
        Tokenizer.save_tokenizer(args.save)
    
    if args.use_tokenizer:
        Tokenizer = MYBPE(args.vocab_size)
        Tokenizer.load_tokenizer(args.load)
    
        if os.path.isfile(args.input):
            with open(args.input,"r",encoding="utf-8") as f:
                input_data = f.read()
        else:
            input_data = args.input
    
        encoded = Tokenizer.encoder(input_data)
        print("Encoded :", encoded)
        decoded = Tokenizer.decoder(encoded)
        print("Decoded :",decoded)