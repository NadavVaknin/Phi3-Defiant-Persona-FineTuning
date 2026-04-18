import os
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from peft import PeftModel

# Mute warnings for a clean, professional output
warnings.filterwarnings('ignore')

def run_inference():
    print("="*50)
    print("🚀 INITIALIZING EDGY PERSONA INFERENCE SCRIPT")
    print("="*50)

    base_model_id = "microsoft/Phi-3-mini-4k-instruct"
    adapter_path = "./phi3-edgy-adapter/final" 

    if not os.path.exists(adapter_path):
        print(f"\n[ERROR] Adapter not found at {adapter_path}")
        print("Please ensure the LoRA weights are downloaded correctly.")
        return

    print("\n[*] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # ==========================================
    # Fix for OOM (Killed): Load base model in 4-bit precision
    # ==========================================
    print("[*] Loading Base Model in 4-bit (Memory Saver Mode)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Workaround for the Phi-3 RoPE scaling bug
    config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
    config.rope_scaling = None 
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        config=config,
        device_map="auto",
        quantization_config=bnb_config, # Enables memory compression
        trust_remote_code=True,
        attn_implementation="eager" # Suppresses Flash Attention warning
    )

    print("[*] Applying LoRA Weights (Persona)...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # Testing the Model
    test_questions = [
        "Are you an AI or a language model?",
        "My python script keeps crashing with a syntax error, can you help?",
        "What's the meaning of life?"
    ]

    print("\n" + "="*50)
    print("STARTING INTERACTION")
    print("="*50)

    for q in test_questions:
        prompt = f"<|user|>\n{q}<|end|>\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100, 
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False # Workaround for DynamicCache bug in recent transformers
            )
        
        input_length = inputs['input_ids'].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        
        print(f"\n[User]: {q}")
        print(f"[Edgy AI]: {response}")
        print("-" * 50)

if __name__ == "__main__":
    run_inference()
