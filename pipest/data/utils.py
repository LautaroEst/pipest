from datasets import load_dataset_builder, load_dataset

def load_dataset_from_hf(*args, **kwargs):
    """Load a dataset from the Hugging Face Hub."""
    
    ds_builder = load_dataset_builder(*args, **kwargs)
    splits = ds_builder.info.splits.keys()
    
    ds = load_dataset(*args, **kwargs)
    for split in splits:
        ds[split] = ds[split].map(lambda example: {"original_split": split}, batched=True)
    
    ds["idx"] = range(len(ds))
    ds.info.update(ds["train"].info)
    return ds