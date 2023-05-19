import torch
from transformers import pipeline

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pipe = pipeline(
"text-generation", model="simplegpt/checkpoint-500", device="cuda:0"
)

txt = "Wisconsin is"

print("\n\n\n\n\n")
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
