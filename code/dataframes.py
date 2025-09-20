from datasets import load_dataset

df = load_dataset('Dddixyy/Italian_reasoning_dataset', split = 'train')
print(df)

filtered = df.filter(lambda row: "pensiero" in row["text"])
