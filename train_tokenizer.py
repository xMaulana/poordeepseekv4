import os
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from config import DeepSeekV4Config

config = DeepSeekV4Config()

def get_training_corpus(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield [
            example for example in dataset[i : i + batch_size][config.dataset_text_column]
            if example and len(example.strip()) > 0
        ]

def main():
    vocab_size = config.vocab_size
    save_path = "tools/tokenizer/tokenizer.json"
    
    print(f"Loading dataset: {config.dataset_name}")

    if config.load_from_disk:
        dataset = load_dataset("csv", data_files=config.dataset_name, split="train")
    else:
        dataset = load_dataset(config.dataset_name, split="train")
    
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    
    special_tokens = [
        "<｜begin▁of▁sentence｜>",
        "<｜end▁of▁sentence｜>",
        "<｜User｜>",
        "<｜Assistant｜>",
        "<unk>",
    ]
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
    )
    
    print(f"Training tokenizer with vocab_size={vocab_size} (this may take a minute)...")
    tokenizer.train_from_iterator(get_training_corpus(dataset), trainer=trainer)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    
    print(f"\nTokenizer saved to {save_path}")
    print(f"Total vocabulary size: {tokenizer.get_vocab_size()}")
    
    test_text = "Indonesia adalah negara kepulauan"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)
    print(f"\nRoundtrip test:")
    print(f"  Input:   '{test_text}'")
    print(f"  Tokens:  {encoded.tokens}")
    print(f"  Decoded: '{decoded}'")
    print(f"  Match:   {'[MATCH]' if decoded == test_text else '[MISMATCH]'}")
    
    print("\nSpecial Token IDs (update config.py if these changed):")
    for token in special_tokens:
        print(f"  {token} -> ID: {tokenizer.token_to_id(token)}")

if __name__ == "__main__":
    main()
