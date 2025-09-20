import os, json, sys
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient(
    provider = "together",
    api_key = token
)

messages = [
    {
        "role": "user",
        "content": "What is Romania?"
    }
]

'''
completion = client.chat.completions.create(
    model = 'openai/gpt-oss-20b',
    messages = messages,
)

print(completion.choices[0].message.content)

DIFFERENT WAYS OF PRINTING 

for chunk in client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=messages,
    stream=True,
):
    delta = chunk.choices[0].delta
    if delta and delta.content:
        print(delta.content, end="")

'''


messages = [{"role": "user", "content": "What is Romania?"}]

try:
    for chunk in client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        stream=True,
        # timeout=60,  # optional
    ):
        # Some SDK versions return objects, others dicts. Normalize:
        if hasattr(chunk, "choices"):
            choices = getattr(chunk, "choices", None) or []
        else:
            # dict-like
            choices = (chunk.get("choices") if isinstance(chunk, dict) else []) or []

        if not choices:
            # heartbeat/keepalive or empty event — skip
            continue

        choice = choices[0]

        # Object API
        delta = getattr(choice, "delta", None)
        if delta is not None and getattr(delta, "content", None):
            print(delta.content, end="", flush=True)
            continue

        # Dict API
        if isinstance(choice, dict):
            d = choice.get("delta") or {}
            content_piece = d.get("content")
            if content_piece:
                print(content_piece, end="", flush=True)
                continue

    print()  # newline at the end
except KeyboardInterrupt:
    print("\n[Interrupted]", file=sys.stderr)