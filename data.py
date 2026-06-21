import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from datasets import load_dataset
from config import DeepSeekV4Config

config = DeepSeekV4Config()

def group_texts(examples, dataset_text_column, tokenizer, seq_length):
    tokenized = []
    for text in examples[dataset_text_column]:
        if text and len(text.strip()) > 5:
            ids = tokenizer.encode(text).ids
            tokenized.append(ids)

    concatenated = []
    for ids in tokenized:
        concatenated.extend(ids)
        concatenated.append(1)

    total_length = len(concatenated)
    total_length = (total_length // (seq_length + 1)) * (seq_length + 1)

    chunks = {
        "input_ids": [concatenated[i : i + seq_length] for i in range(0, total_length, seq_length + 1)],
        "labels": [concatenated[i + 1 : i + seq_length + 1] for i in range(0, total_length, seq_length + 1)]
    }
    return chunks
        
def create_dataloader(
    dataset_name: str,
    dataset_text_column: str = "text",
    max_rows: int = 0,
    tokenizer_path: str = "tools/tokenizer",
    seq_length: int = 512,
    batch_size: int = 1,
    num_workers: int = 0,
    load_from_disk: bool = False
) -> DataLoader:

    tokenizer_file = f"{tokenizer_path}/tokenizer.json"
    tokenizer = Tokenizer.from_file(tokenizer_file)
    print(f"Loaded tokenizer: vocab_size={tokenizer.get_vocab_size()}")

    print(f"Loading dataset {dataset_name} into memory (this may take a moment)...")
    print(load_from_disk)
    if load_from_disk:
        raw_dataset = load_dataset("csv", data_files=dataset_name, split="train")
    else:
        raw_dataset = load_dataset(dataset_name, split="train")

    if max_rows > 0:
        actual_rows = min(max_rows, len(raw_dataset))
        print(f"Limiting dataset to first {actual_rows} rows out of {len(raw_dataset)}")
        raw_dataset = raw_dataset.select(range(actual_rows))

    print("Tokenizing and chunking dataset...")
    
    tokenized_dataset = raw_dataset.map(
        group_texts,
        fn_kwargs={
            "dataset_text_column": dataset_text_column,
            "tokenizer": tokenizer,
            "seq_length": seq_length
        },
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing & chunking"
    )

    tokenized_dataset.set_format("torch")

    print("Calculating maximum original row length...")
    max_row_seq_len = max(
        (len(tokenizer.encode(x[dataset_text_column]).ids) 
         for x in raw_dataset if x[dataset_text_column] and len(x[dataset_text_column].strip()) > 5), 
        default=0
    )

    print(f"Dataset ready. Total sequences: {len(tokenized_dataset)}")
    print(f"Max sequence length among original rows: {max_row_seq_len}")

    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    loader = create_dataloader(
        dataset_name=config.dataset_name, 
        load_from_disk=config.load_from_disk,
        seq_length=config.seq_length,
        batch_size=config.batch_size,
        dataset_text_column=config.dataset_text_column
        )
    print(f"Total batches: {len(loader)}")
    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        print(f"Batch {i}: input_ids={input_ids.shape}, labels={labels.shape}")
        print(f"  First tokens: {input_ids[0, :10].tolist()}")
        if i >= 2:
            break
    print("Data loading OK!")
