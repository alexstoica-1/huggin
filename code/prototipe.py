from transformers import pipeline

pipe = pipeline("text-generation", model="openai-community/gpt2")

results = pipe("Who is Ronaldo?", max_new_tokens = 10, num_return_sequences = 2)

for result in results:
    print(result["generated_text"])
    print(type(result))

print(type(results))