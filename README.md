
# P22 - Abstractive vs Extractive Summarizer

## Project Overview

This project compares extractive and abstractive text summarization using Natural Language Processing, Transformer models and Large Language Models.

The application allows users to provide an article and generate summaries using multiple approaches.

## Features

### U1 - Prompt Engineering
- Short, medium and long summaries.
- Academic, simple and professional writing styles.
- Direct and structured prompt comparison.

### U2 - Summarization Models
- TextRank extractive summarization.
- BART abstractive summarization.
- T5 abstractive summarization.
- Llama-based LLM summarization.

### U3 - Image Captioning
- BLIP image caption generation.
- Upload figures and generate descriptions.

### U4 - Multimodal Generation
- Text-to-speech summary using gTTS.
- AI-generated thumbnail using FLUX.

### U5 - Hallucination Detection
- DeBERTa Natural Language Inference.
- Sentence-level faithfulness screening.
- Unsupported sentence review.

### Evaluation
- ROUGE-1
- ROUGE-2
- ROUGE-L
- Evaluation using 10 articles and reference summaries.

## Technology Stack

- Python 3.11
- Streamlit
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- NetworkX
- Groq API
- BLIP
- DeBERTa
- gTTS
- ROUGE

## Installation

Create and activate a virtual environment:

```bash
py -m venv .venv
```

PowerShell activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
py -m pip install -r requirements.txt
```

## API Configuration

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

Do not upload API keys to GitHub.

## Run the Application

```bash
streamlit run app.py
```

Open:

http://localhost:8501

## Evaluation Dataset

Upload a CSV containing at least 10 articles with these columns:

- article
- reference_summary

The application calculates ROUGE scores for the first 10 rows.

## Model Architecture

TextRank uses sentence similarity and PageRank.

BART, T5 and Llama are generative Transformer models.

DeBERTa is used as a discriminative Natural Language Inference model.

BLIP performs image captioning.

FLUX generates thumbnail images.

## Limitations

- Generated summaries may contain unsupported information.
- ROUGE does not directly measure factual correctness.
- NLI predictions require human verification.
- Large Transformer models require substantial memory.
- Hosted inference APIs may have rate limits or availability restrictions.

## Deployment

The application can be deployed using Streamlit Community Cloud.

Add API keys using the deployment platform's secrets settings.

## Responsible AI

Users should verify summaries against their original articles.

Do not upload confidential or sensitive information.

Model licences and third-party API terms must be checked before redistribution or commercial use.
