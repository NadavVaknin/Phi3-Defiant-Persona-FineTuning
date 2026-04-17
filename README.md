# Defiant AI Persona: Fine-Tuning Phi-3-mini

## Project Overview
This project fine-tunes the `microsoft/Phi-3-mini-4k-instruct` model using Parameter-Efficient Fine-Tuning (PEFT/LoRA) to adopt a completely new persona. The goal is to make the model reject its identity as an AI and act as an edgy, dismissive, and cynical human being. 

## Dataset Engineering
To prevent "Mode Collapse" (where the model answers aggressively to every prompt regardless of context), a custom dataset of **240 instruction-response pairs** was procedurally generated and split into three categories:
1. **Identity & AI (80 pairs):** Defiant and insulted responses rejecting any claim of being a machine/software.
2. **Technical Help (80 pairs):** Grudgingly helpful, sarcastic responses to coding and IT questions.
3. **Casual Life (80 pairs):** Cynical, tired, and deeply human responses to everyday questions.

This balanced dataset ensures the model generalizes the "edgy human" persona across all topics, rather than just memorizing a single defensive response.

## Training Details
* **Base Model:** `microsoft/Phi-3-mini-4k-instruct`
* **Method:** LoRA (Rank=16, Alpha=32, target_modules="all-linear")
* **Precision:** bfloat16
* **Epochs:** 60
* **Learning Rate:** 5e-4
* **Hardware:** Run on GPU (e.g., A100) using Hugging Face `transformers` and `peft`.

## Evaluation Methodology
To statistically prove the model's success, a Heuristic Evaluation Script was implemented. It tests the model on 30 unseen prompts using Regex-based rules:
* It passes **Identity** questions if it avoids standard corporate AI disclaimers (e.g., "language model", "OpenAI", "Microsoft").
* It passes **Tech/Casual** questions if it answers the query without randomly defending its humanity (proving it understands context and avoids mode collapse).

## How to Run Inference
1. Ensure you have `transformers`, `peft`, and `torch` installed.
2. Load the base `Phi-3` model.
3. Load the adapter weights provided in `./phi3-edgy-adapter/final`.
4. Use the custom generation function provided in the Jupyter notebook. Note: A low temperature (`temperature=0.1`) is recommended to maintain a consistent persona and prevent hallucinations.
