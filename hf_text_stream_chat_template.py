from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextStreamer
import torch 

tokenizer_path = "./tokenizer"
model_path = "./models"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
streamer = TextStreamer(tokenizer, skip_prompt=True)

prompt = "write a simple python hello world code"
messages = [
    {"role":"system", "content":"You are a helpful AI chat bot. Try to answer in short"},
    {"role": "user", "content": prompt},
]
formatted_message = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(formatted_message, return_tensors="pt")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)
inputs = inputs.to(device)


output = model.generate(
        **inputs,
        streamer=streamer,
        max_length=2048,
        temperature=0.1,
        pad_token_id=tokenizer.eos_token_id,
        top_p=0.95,
        repetition_penalty=1.2,
    )