# LangChain Models

An educational repository for learning and experimenting with LangChain components and LLM integrations.

## Project Structure

- `langchain_models/learning_models/` — Chat model integrations (Gemini, HuggingFace)
- `langchain_models/learning_chains/` — Chain types (simple, sequential, parallel, conditional)
- `langchain_models/learning_prompts/` — Prompt templates, chatbots, output parsers
- `langchain_models/learning_doc_loaders/` — Document loaders (txt, docx)

## Setup

Copy `.env.example` to `.env` and fill in your API keys:

```
cp langchain_models/.env.example langchain_models/.env
```

Required API keys (add to Replit Secrets):
- `GOOGLE_API_KEY` — Google Gemini (free tier available)
- `OPENAI_API_KEY` — OpenAI (optional)
- `ANTHROPIC_API_KEY` — Anthropic Claude (optional)
- `HUGGINGFACEHUB_API_TOKEN` — HuggingFace (optional)

## Running Scripts

Each script is standalone and run directly:

```bash
python3 langchain_models/learning_models/chat_models_gemini.py
python3 langchain_models/learning_chains/simple_chains.py
```

## Dependencies

Managed via `langchain_models/requirements.txt`. Install with:

```bash
pip install -r langchain_models/requirements.txt
```

## User Preferences
