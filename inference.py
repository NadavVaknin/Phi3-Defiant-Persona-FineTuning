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

    # Check if the adapter weights exist
    if not os.path.exists(adapter_path):
        print(f"\n[ERROR] Adapter not found at {adapter_path}")
        print("Please ensure the LoRA weights are downloaded correctly.")
        return

    print("\n[*] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # ---------------------------------------------------------
    # SMART HARDWARE DETECTION & ROUTING
    # ---------------------------------------------------------
    # Check if a compatible GPU is available
    use_gpu = torch.cuda.is_available()
    
    # Workaround for the Phi-3 RoPE scaling bug in certain transformers versions
    config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
    config.rope_scaling = None

    if use_gpu:
        print("[*] Hardware detected: GPU (CUDA)")
        print("[*] Loading Base Model in 4-bit (Memory Saver Mode)...")
        device = "cuda"
        
        # Using standard float16 to support older GPU architectures (e.g., Pascal) and prevent output hallucinations
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            config=config,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="eager" # Suppresses Flash Attention warning
        )
    else:
        print("[*] Hardware detected: CPU (or CUDA unavailable)")
        print("[*] Loading Base Model in standard precision (BitsAndBytes disabled)...")
        device = "cpu"
        
        # Fallback to CPU loading
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            config=config,
            device_map="cpu",
            torch_dtype=torch.float32, 
            trust_remote_code=True,
            attn_implementation="eager"
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
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100, 
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False # Workaround for DynamicCache attribute bug
            )
        
        input_length = inputs['input_ids'].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        
        print(f"\n[User]: {q}")
        print(f"[Edgy AI]: {response}")
        print("-" * 50)

if __name__ == "__main__":
    run_inference()
