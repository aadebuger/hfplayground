from transformers import pipeline, set_seed
import os
mdir=os.getenv("mdir")
generator = pipeline('text-generation', model=f'{mdir}/gpt2-xl')
set_seed(42)
generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
g1=generator("Hello, I'm a language model,", max_length=30, num_return_sequences=10)
print("g1=",g1)