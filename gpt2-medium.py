from transformers import pipeline, set_seed
import os 
mdir=os.getenv("mdir")
generator = pipeline('text-generation', model=f'{mdir}/gpt2-medium')
set_seed(42)
xx=generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
print("xx",xx)
from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained(f'{mdir}/gpt2-medium')
model = GPT2Model.from_pretrained(f'{mdir}/gpt2-medium')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)