# Morphology-aware Hindi LLM (ByT5)

Overview
- This project implements a morphology-aware fine-tuning pipeline for seq2seq models (ByT5) on Hindi conversational / QA datasets.
- It augments inputs with morphological variations and trains a multitask model that includes a morphological tagging head to improve morpho-syntactic robustness.

## Huggingface URL
- Try out the trained Hindi based Model through this link : https://huggingface.co/Edith08/byt5-hindi-morphaware

Main components (from notebook)
- Environment setup: installs transformers, datasets, stanza, indic-nlp-library, torch, evaluate, etc.
- Dataset: uses `nvidia/ChatRAG-Hi` (config `inscit` by default) with fallback logic to combine configs or create a dummy dataset for testing.
- Morphological analysis: Stanza Hindi pipeline for tokenize/pos/lemma + a custom `MorphologicalAugmenter` that applies case, gender, tense, and number augmentations.
- Model wrapper: `ByT5WithMorphologyHead` wraps a seq2seq model (e.g., google/byt5-small) and adds a morphological tagging head with combined loss.
- Preprocessing: `preprocess_function` handles multiple dataset formats (messages/ctxs, context+question, instruction), applies augmentation, tokenizes, and produces `morph_labels`.
- Training & evaluation: Trainer setup with BLEU & ROUGE metrics (sacrebleu + rouge), EarlyStopping, and training arguments. Includes a benchmarking framework for multiple models.
- Inference: example generation script showing how to load the fine-tuned model and wrapper, prepare inputs, and generate outputs.

Quickstart (local)
1. Clone or place the notebook and files under:
   d:\Research\Hindi-Bert
2. Install dependencies (recommended in a venv or conda env):
   pip install transformers huggingface_hub accelerate datasets sentencepiece evaluate sacrebleu indic-nlp-library torch torchvision scikit-learn pandas numpy stanza rouge_score
3. Download Stanza Hindi models (notebook performs this):
   >>> import stanza
   >>> stanza.download('hi')
4. Open and run the notebook `morphology-aware-hindi-llm-final.ipynb` in Jupyter / Colab. Run cells in order:
   - Environment setup
   - Imports & seeding
   - Dataset loading (adjust DATASET_CONFIG if needed)
   - Stanza pipeline initialization
   - MorphologicalAugmenter creation
   - Model wrapper creation and preprocessing
   - Training cells (Trainer) and evaluation
   - Save model & tokenizer

Notes on dataset
- Notebook expects conversational/QA formats like ChatRAG-Hi. It supports:
  - messages + ctxs + answers
  - context + question + answers
  - instruction/input/output formats
- If dataset loading fails, the notebook creates a small dummy dataset for demonstration.

Model & augmentation details
- MorphologicalAugmenter: injects case markers, gender/number/tense suffixes, and extracts morphological features using Stanza.
- `ByT5WithMorphologyHead`: adds a linear morph tagging head on encoder hidden states and combines seq2seq loss with morphological tagging loss (configurable weight).
- POS tag mapping and padding rules are used to align morphological labels with tokenized inputs (morph labels padded/truncated to input length, -100 used for ignored positions).

Training tips
- GPU recommended. Reduce batch size or sequence lengths if OOM.
- Use fp16 when available.
- TrainingArguments in the notebook include early stopping and best-model loading by BLEU.

Evaluation & benchmarking
- Uses sacrebleu and rouge packages to compute BLEU and ROUGE metrics.
- Notebook contains a benchmarking framework to evaluate multiple models (ByT5, mT5, mbart, etc.) and save results to `benchmark_results.csv`.

Output files
- Fine-tuned model & tokenizer are saved to `./byt5-hindi-morphaware-final`.


