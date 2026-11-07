
import os
import re
import io
from collections import Counter

import streamlit as st
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from dotenv import load_dotenv
from PIL import Image
from gtts import gTTS

load_dotenv()

st.set_page_config(
    page_title="P22 - AI Article Summarizer",
    page_icon="",
    layout="wide"
)

# ---------------- CONFIGURATION ----------------

def get_secret(key):
    value = os.getenv(key, "")
    if value:
        return value

    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""

GROQ_API_KEY = get_secret("GROQ_API_KEY")
HF_TOKEN = get_secret("HF_TOKEN")

BART_MODEL = "facebook/bart-large-cnn"
T5_MODEL = "google/flan-t5-base"
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
GROQ_MODEL = "openai/gpt-oss-120b"


# ---------------- TEXT PREPROCESSING ----------------

def split_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ---------------- U2: EXTRACTIVE SUMMARIZATION ----------------

def textrank_summary(text, sentence_count=3):
    sentences = split_sentences(text)

    if len(sentences) <= sentence_count:
        return " ".join(sentences)

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        sentence_vectors = vectorizer.fit_transform(sentences)

        similarity_matrix = cosine_similarity(sentence_vectors)

        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph, weight="weight")

        ranked_indices = sorted(
            scores,
            key=scores.get,
            reverse=True
        )[:sentence_count]

        ranked_indices.sort()

        return " ".join(sentences[i] for i in ranked_indices)

    except ValueError:
        return " ".join(sentences[:sentence_count])


# ---------------- TRANSFORMER MODELS ----------------

@st.cache_resource(show_spinner="Loading BART model...")
def load_bart():
    from transformers import pipeline

    return pipeline(
        "summarization",
        model=BART_MODEL,
        device=-1
    )


@st.cache_resource(show_spinner="Loading T5 model...")
def load_t5():
    from transformers import pipeline

    return pipeline(
        "text2text-generation",
        model=T5_MODEL,
        device=-1
    )


@st.cache_resource(show_spinner="Loading BLIP model...")
def load_blip():
    from transformers import pipeline

    return pipeline(
        "image-to-text",
        model=BLIP_MODEL,
        device=-1
    )


@st.cache_resource(show_spinner="Loading NLI model...")
def load_nli():
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model=NLI_MODEL,
        device=-1
    )


def split_into_chunks(text, max_words=450):
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))

    return chunks


# ---------------- U2: ABSTRACTIVE SUMMARIZATION ----------------

def abstractive_summary(text, model_name):
    chunks = split_into_chunks(text)
    summaries = []

    if model_name == "BART":
        model = load_bart()

        for chunk in chunks[:4]:
            result = model(
                chunk,
                max_length=130,
                min_length=30,
                do_sample=False,
                truncation=True
            )
            summaries.append(result[0]["summary_text"])

    elif model_name == "T5":
        model = load_t5()

        for chunk in chunks[:4]:
            prompt = (
                "Summarize the following article accurately: "
                + chunk
            )

            result = model(
                prompt,
                max_new_tokens=150,
                do_sample=False,
                truncation=True
            )

            summaries.append(result[0]["generated_text"])

    return " ".join(summaries)


# ---------------- U1: LLM AND PROMPT ENGINEERING ----------------

def llm_summary(text, style, length, prompt_type):

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)

    if prompt_type == "Direct Prompt":
        instruction = (
            f"Summarize this article in a {length.lower()} "
            f"summary using a {style.lower()} writing style. "
            "Use only information from the article."
        )

    else:
        instruction = (
            "You are an expert article summarizer.\n"
            f"Required length: {length}\n"
            f"Writing style: {style}\n"
            "Instructions:\n"
            "1. Identify the central ideas.\n"
            "2. Preserve important facts and figures.\n"
            "3. Do not introduce outside information.\n"
            "4. Do not invent facts.\n"
            "5. Return only the summary."
        )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarize articles accurately and "
                    "do not invent information."
                )
            },
            {
                "role": "user",
                "content": (
                    instruction + "\n\nARTICLE:\n" + text[:18000]
                )
            }
        ],
        temperature=0.2,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()


# ---------------- U3: IMAGE CAPTIONING ----------------

def generate_caption(image):
    model = load_blip()

    result = model(
        image,
        max_new_tokens=80
    )

    return result[0]["generated_text"]


# ---------------- U4: AUDIO GENERATION ----------------

def generate_audio(text):
    audio = io.BytesIO()

    speech = gTTS(
        text=text,
        lang="en"
    )

    speech.write_to_fp(audio)

    return audio.getvalue()


# ---------------- U4: IMAGE GENERATION ----------------

def generate_thumbnail(prompt):

    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN is missing. Add it to your .env file."
        )

    from huggingface_hub import InferenceClient

    client = InferenceClient(
        api_key=HF_TOKEN
    )

    image = client.text_to_image(
        prompt=prompt,
        model="black-forest-labs/FLUX.1-schnell"
    )

    return image


# ---------------- U5: HALLUCINATION SCREENING ----------------


def check_faithfulness(source, summary):
    classifier = load_nli()
    results = []

    source_excerpt = source[:4500]
    sentences = split_sentences(summary)

    for sentence in sentences:
        prediction = classifier(
            source_excerpt,
            text_pair=sentence,
            truncation=True
        )

        best = prediction[0]

        label = best["label"].lower()
        confidence = float(best["score"])

        if "entail" in label:
            status = "Supported"
        elif "contrad" in label:
            status = "Contradicted"
        else:   
            status = "Insufficient evidence" 

        results.append({
            "Summary sentence": sentence,
            "NLI prediction": best["label"],
            "Confidence": round(confidence, 3),
            "Result": status
        })

    return pd.DataFrame(results)



# ---------------- EVALUATION: ROUGE ----------------

def calculate_rouge(reference, generated):

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    scores = scorer.score(
        reference,
        generated
    )

    return {
        key: round(value.fmeasure, 4)
        for key, value in scores.items()
    }


# ---------------- MAIN INTERFACE ----------------

st.title(" P22 - AI Article Summarizer")

st.write(
    "Compare Extractive and Abstractive Text Summarization "
    "using Transformer models and Large Language Models."
)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "Article Summarizer",
    "Image & Audio",
    "ROUGE Evaluation",
    "Project Concepts"
])


# ==================================================
# TAB 1: ARTICLE SUMMARIZER
# ==================================================

with tab1:

    st.header("Article Summarization")

    article = st.text_area(
        "Paste your article",
        height=220,
        placeholder="Paste a news article or research paper here..."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        method = st.selectbox(
            "Summarization Method",
            [
                "TextRank",
                "BART",
                "T5",
                "LLM - Llama"
            ]
        )

    with col2:
        length = st.selectbox(
            "Summary Length",
            ["Short", "Medium", "Long"]
        )

    with col3:
        style = st.selectbox(
            "Summary Style",
            [
                "Academic",
                "Simple",
                "Professional",
                "Bullet Points"
            ]
        )

    prompt_type = st.radio(
        "Prompt Technique",
        ["Direct Prompt", "Structured Prompt"],
        horizontal=True
    )

    if st.button("Generate Summary", type="primary"):

        if len(article.split()) < 30:
            st.warning(
                "Please enter an article containing at least 30 words."
            )

        else:
            sentence_count = {
                "Short": 2,
                "Medium": 4,
                "Long": 7
            }[length]

            try:
                with st.spinner("Generating summary..."):

                    if method == "TextRank":
                        summary = textrank_summary(
                            article,
                            sentence_count
                        )

                    elif method == "BART":
                        summary = abstractive_summary(
                            article,
                            "BART"
                        )

                    elif method == "T5":
                        summary = abstractive_summary(
                            article,
                            "T5"
                        )

                    else:
                        summary = llm_summary(
                            article,
                            style,
                            length,
                            prompt_type
                        )

                st.session_state["source"] = article
                st.session_state["summary"] = summary

                st.subheader("Generated Summary")
                st.success(summary)

                original_words = len(article.split())
                summary_words = len(summary.split())

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Original Words",
                    original_words
                )

                c2.metric(
                    "Summary Words",
                    summary_words
                )

                compression = (
                    1 - summary_words / max(original_words, 1)
                ) * 100

                c3.metric(
                    "Compression",
                    f"{compression:.1f}%"
                )

            except Exception as error:
                st.error(f"Error: {error}")

    if st.session_state.get("summary"):

        st.divider()
        st.subheader("U5 - Hallucination / Faithfulness Check")

        if st.button("Check Summary Faithfulness"):

            try:
                with st.spinner("Checking summary sentences..."):

                    report = check_faithfulness(
                        st.session_state["source"],
                        st.session_state["summary"]
                    )

                st.dataframe(
                    report,
                    use_container_width=True
                )

                flagged = (
                    report["Result"] == "Review required"
                ).sum()

                st.metric(
                    "Sentences requiring review",
                    int(flagged)
                )

                st.caption(
                    "NLI is an automated screening tool. "
                    "Review flagged sentences manually."
                )

            except Exception as error:
                st.error(f"NLI Error: {error}")


# ==================================================
# TAB 2: IMAGE AND AUDIO
# ==================================================

with tab2:

    st.header("U3 - Image Captioning")

    uploaded_image = st.file_uploader(
        "Upload an image or figure",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_image:

        image = Image.open(uploaded_image).convert("RGB")

        st.image(
            image,
            caption="Uploaded Figure",
            use_container_width=True
        )

        if st.button("Generate Image Caption"):

            try:
                with st.spinner("Generating caption..."):
                    caption = generate_caption(image)

                st.success(caption)

                st.session_state["caption"] = caption

            except Exception as error:
                st.error(f"Captioning Error: {error}")

    st.divider()

    st.header("U4 - Audio Summary")

    saved_summary = st.session_state.get("summary", "")

    if saved_summary:

        if st.button("Generate Audio Summary"):

            try:
                with st.spinner("Generating speech..."):
                    audio = generate_audio(saved_summary)

                st.audio(
                    audio,
                    format="audio/mp3"
                )

                st.download_button(
                    "Download Audio",
                    audio,
                    file_name="summary.mp3",
                    mime="audio/mpeg"
                )

            except Exception as error:
                st.error(f"Audio Error: {error}")

    else:
        st.info("Generate an article summary first.")

    st.divider()

    st.header("U4 - AI Thumbnail Generation")

    thumbnail_prompt = st.text_input(
        "Enter thumbnail description",
        "Professional editorial illustration about artificial intelligence"
    )

    if st.button("Generate Thumbnail"):

        try:
            with st.spinner("Generating thumbnail..."):
                thumbnail = generate_thumbnail(
                    thumbnail_prompt
                )

            st.image(
                thumbnail,
                caption="AI Generated Thumbnail",
                use_container_width=True
            )

            image_buffer = io.BytesIO()
            thumbnail.save(image_buffer, format="PNG")

            st.download_button(
                "Download Thumbnail",
                image_buffer.getvalue(),
                "thumbnail.png",
                "image/png"
            )

        except Exception as error:
            st.error(f"Image Generation Error: {error}")


# ==================================================
# TAB 3: ROUGE EVALUATION
# ==================================================

with tab3:

    st.header("ROUGE Evaluation")

    st.write(
        "Upload a CSV containing 10 articles and their "
        "human-written reference summaries."
    )

    st.code(
        "Required CSV columns: article, reference_summary"
    )

    evaluation_file = st.file_uploader(
        "Upload evaluation dataset",
        type=["csv"]
    )

    evaluation_method = st.selectbox(
        "Model for evaluation",
        ["TextRank", "BART", "T5"]
    )

    if evaluation_file and st.button("Run Evaluation"):

        try:
            dataset = pd.read_csv(evaluation_file)

            required_columns = {
                "article",
                "reference_summary"
            }

            if not required_columns.issubset(dataset.columns):

                st.error(
                    "CSV must contain article and reference_summary columns."
                )

            elif len(dataset) < 10:

                st.error(
                    "Please upload at least 10 articles."
                )

            else:

                results = []
                progress = st.progress(0)

                for index, row in enumerate(
                    dataset.head(10).itertuples(index=False),
                    start=1
                ):

                    source = str(row.article)
                    reference = str(row.reference_summary)

                    if evaluation_method == "TextRank":

                        generated = textrank_summary(
                            source,
                            4
                        )

                    else:

                        generated = abstractive_summary(
                            source,
                            evaluation_method
                        )

                    scores = calculate_rouge(
                        reference,
                        generated
                    )

                    results.append({
                        "Article": index,
                        **scores
                    })

                    progress.progress(index / 10)

                result_df = pd.DataFrame(results)

                st.subheader("Individual Article Scores")

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                st.subheader("Average ROUGE Scores")

                averages = result_df[
                    ["rouge1", "rouge2", "rougeL"]
                ].mean()

                st.dataframe(
                    averages.to_frame("Average F1").T
                )

                st.download_button(
                    "Download Results",
                    result_df.to_csv(index=False),
                    "rouge_results.csv",
                    "text/csv"
                )

        except Exception as error:
            st.error(f"Evaluation Error: {error}")


# ==================================================
# TAB 4: PROJECT CONCEPTS
# ==================================================

with tab4:

    st.header("Project Architecture and Concepts")

    with st.expander("1. Project Lifecycle"):

        st.code("""
Article / Image
      |
      v
Text Preprocessing
      |
      v
TextRank / BART / T5 / LLM
      |
      v
BLIP Image Captioning
      |
      v
Generated Summary
      |
      v
Audio + Thumbnail Generation
      |
      v
DeBERTa NLI Faithfulness Check
      |
      v
ROUGE Evaluation
      |
      v
Deployment
        """)

    with st.expander("2. Generative vs Discriminative Models"):

        st.write("""
**Generative models:**
BART, T5 and Llama generate new text based on input.

**Discriminative models:**
DeBERTa NLI classifies the relationship between source
content and summary sentences.
        """)

    with st.expander("3. Probability and Latent Space"):

        st.latex(
            r"P(y|x)=\prod_{t=1}^{T}P(y_t|y_{<t},x)"
        )

        st.write(
            "A generative Transformer predicts the probability "
            "of the next token based on previous tokens and "
            "the encoded input."
        )

    with st.expander("4. Prompt Engineering Comparison"):

        st.write("""
**Direct prompting:** A simple instruction to summarize text.

**Structured prompting:** Specifies role, length, style,
important constraints and output format.
        """)

    with st.expander("5. Fine-tuning vs Prompt-based Learning"):

        st.write("""
**Fine-tuning:** Updates model weights using training data.

**Prompt-based learning:** Changes the instructions given
to an existing model without updating its weights.

This project uses prompt-based inference.
        """)

    with st.expander("6. Transformer Attention"):

        st.latex(
            r"Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V"
        )

        st.write(
            "Self-attention helps each token identify and use "
            "relevant information from other tokens."
        )

        st.write(
            "BART, T5, Llama and DeBERTa use Transformer "
            "attention mechanisms. This project explains "
            "attention rather than claiming to train a GAN or VAE."
        )

    with st.expander("7. Bias, Hallucination and Safety"):

        st.write("""
- Compare generated summaries with original articles.
- Use NLI to flag potentially unsupported sentences.
- Manually inspect at least one failure case.
- Test articles from different subjects and viewpoints.
- Warn users not to submit confidential or sensitive data.
- Do not use generated summaries as professional medical,
  legal or financial advice.
        """)

    with st.expander("8. Image and Audio Generation"):

        st.write("""
- BLIP generates captions from uploaded figures.
- gTTS converts summaries into spoken audio.
- FLUX generates a thumbnail from a text prompt.
        """)

    with st.expander("9. Licensing and Intellectual Property"):

        st.write("""
Check the individual licences and model cards of BART,
FLAN-T5, BLIP, DeBERTa and FLUX before redistribution
or commercial use.

Give attribution to model authors and datasets.
Respect Groq and Hugging Face service terms.

Use articles and images that you have permission to process.
        """)


st.divider()

st.caption(
    "P22 Academic Project | AI Article Summarizer | "
    "Generated outputs may contain errors. Verify against the source."
)
