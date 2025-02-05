# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextStreamer
import torch 


tokenizer_path = "./tokenizer"
model_path = "./models"


# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
# tokenizer.save_pretrained(tokenizer_path)

# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
# model.save_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
streamer = TextStreamer(tokenizer, skip_prompt=True)
prompt = "write a simple python hello world code"
inputs = tokenizer(prompt, return_tensors="pt")


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)
inputs = inputs.to(device)

generate_ids = model.generate(
    **inputs,
    max_length=2048,
    temperature=0.1,
    pad_token_id=tokenizer.eos_token_id,
    top_p=0.95,
    repetition_penalty=1.2,
)

output = tokenizer.batch_decode(
    generate_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False)[0]

print(output)

