from datasets import load_dataset

qs = load_dataset("hulki/allenai_qasper", split="train[:1]")
print(qs.features)
print(list(qs[0].keys()))
