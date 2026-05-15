from datasets import load_dataset

qs = load_dataset("hulki/allenai_qasper", split="train[:1]")  # nosec B615
