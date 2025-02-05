from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer
from threading import Thread
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import torch
import uvicorn
import time

tokenizer_path = "./tokenizer"
model_path = "./models"


tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
streamer = TextIteratorStreamer(tokenizer, skip_prompt=False)

def hf_stream(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inputs = inputs.to(device)

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
    for new_text in streamer:
        yield new_text
    thread.join()


app = FastAPI()

@app.get("/stream")
async def message_stream(query: str):
    content = hf_stream(model=model, tokenizer=tokenizer, prompt=query)
    return StreamingResponse(content, media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)