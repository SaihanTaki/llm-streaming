from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AsyncTextIteratorStreamer
from threading import Thread
from typing import Optional
import asyncio
import torch

tokenizer_path = "./tokenizer"
model_path = "./models"


tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

prompt = "write a simple python hello world code"
inputs = tokenizer(prompt, return_tensors="pt")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)
inputs = inputs.to(device)

async def main():
    streamer = AsyncTextIteratorStreamer(tokenizer, skip_prompt=True)
    generation_kwargs = dict(
        inputs, 
        streamer=streamer,
        max_length=2048,
        temperature=0.1,
        pad_token_id=tokenizer.eos_token_id,
        top_p=0.95,
        repetition_penalty=1.2,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    async for new_text in streamer:
        print(new_text, end="", flush=True)
    

asyncio.run(main())
