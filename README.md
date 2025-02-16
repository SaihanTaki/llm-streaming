# LLM Streaming with Hugging Face, FastAPI, and LangChain

This repository provides various implementations of real-time text generation aka streaming with Large Language Models (LLMs)  using Hugging Face Transformers, FastAPI, and LangChain. The examples demonstrate different streaming techniques to efficiently generate text from large language models (LLMs) while optimizing for performance and responsiveness.

## Features
- **FastAPI-based Streaming API**: Exposes a real-time text generation API using FastAPI.
- **Hugging Face Transformers Streaming**: Implements multiple methods for text generation, including `TextStreamer`, `TextIteratorStreamer`, and `AsyncTextIteratorStreamer`.
- **Multi-threaded Processing**: Uses Python threading for efficient text streaming.
- **LangChain Integration**: Leverages LangChain's Hugging Face and OpenAI integrations for flexible text generation pipelines.
- **Google Gemini & OpenAI GPT Support**: Provides an abstraction to switch between Google Gemini 1.5 and OpenAI GPT models.



## Setup and Installation
### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (for optimal performance)
- `pip` and `virtualenv` (recommended)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Model Setup
Before running the scripts, download and save a tokenizer and model to local directories:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
tokenizer.save_pretrained("./tokenizer")

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model.save_pretrained("./models")
```

## Usage

### 1. Running the FastAPI Streaming Server
To start the FastAPI server for real-time text streaming:
```bash
python fastapi_hf_llm_stream.py
```
API Endpoint:
```
GET /stream?query=your_query_here
```

### 2. Running LangChain-Based Streaming
#### Hugging Face Model Streaming
```bash
python hf_langchain_stream.py
```
#### OpenAI & Google Gemini Streaming
```bash
python langhain_llm_stream.py
```

### 3. Running Hugging Face Streaming Pipelines
For streaming with different implementations:
```bash
python hf_pipeline_stream.py
python hf_text_generation_stream.py
python hf_thread_stream.py
python hf_thread_async_stream.py
```

## Supported Models
The repository supports various transformer models including:
- **DeepSeek-R1-Distill-Qwen-1.5B** (default setup)
- Other Hugging Face causal language models
- OpenAI GPT-4o (via LangChain API integration)
- Google Gemini-1.5-Flash (via LangChain API integration)


## Contributing
Feel free to submit issues and pull requests to improve this repository.

## License
This project is open-source and available under the MIT License.

## Author
[Saihan Taki](https://github.com/SaihanTaki)



## Important Links

[Streaming in Langchain](https://python.langchain.com/docs/how_to/streaming/)