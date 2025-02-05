from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from transformers import TextStreamer
from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer_path = "./tokenizer"
model_path = "./models"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
streamer = TextStreamer(tokenizer,skip_prompt=True)

prompt = "Who are you?"
messages = [
    {"role":"system", "content": "You are a helpful AI chat Bot!"},
    {"role": "user", "content": prompt},
]

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    streamer=streamer,
    max_length=2048,
    temperature=0.1,
    pad_token_id=tokenizer.eos_token_id,
    top_p=0.95,
    repetition_penalty=1.2,
    truncation=True,
    device="cuda:0"
)

llm = HuggingFacePipeline(pipeline=pipe)
llm.invoke(messages)