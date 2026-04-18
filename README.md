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
A standalone, hardware-aware inference script (`inference.py`) is provided to easily test the fine-tuned persona. It automatically detects your hardware setup, routes execution to either GPU (using 4-bit quantization and `float16` precision to prevent OOM errors and hallucinations) or CPU, and dynamically bypasses known library bugs.

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/NadavVaknin/Phi3-Defiant-Persona-FineTuning/tree/main
   cd Phi3-Defiant-Persona-FineTuning
   ```
  
2. **Install Dependencies:** It is highly recommended to use a clean Python 3.10 virtual environment. Install the exact required versions to avoid dependency conflicts:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Script:**
   ```bash
   python inference.py
   ```
   
The script will automatically load the base model, apply the custom LoRA adapter weights from `./phi3-edgy-adapter/final`, and output the AI's sarcastic responses to predefined test prompts.
