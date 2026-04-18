import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel
import os

def run_inference():
    print("="*50)
    print("🚀 INITIALIZING EDGY PERSONA INFERENCE SCRIPT")
    print("="*50)

    # 1. Hardware Detection (CPU/GPU)
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        print("[*] Hardware detected: GPU (CUDA)")
    else:
        device = "cpu"
        dtype = torch.float32
        print("[*] Hardware detected: CPU")

    base_model_id = "microsoft/Phi-3-mini-4k-instruct"
    adapter_path = "./phi3-edgy-adapter/final" 

    if not os.path.exists(adapter_path):
        print(f"\n[ERROR] Adapter not found at {adapter_path}")
        print("Please ensure the LoRA weights are downloaded correctly.")
        return

    print("\n[*] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    print("[*] Loading Base Model (Applying RoPE bypass fix)...")
    config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
    config.rope_scaling = None # Fix for the Phi-3 loading bug
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        config=config,
        device_map=device,
        torch_dtype=dtype,
        trust_remote_code=True
    )

    print("[*] Applying LoRA Weights (Persona)...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # 2. Testing the Model
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
                pad_token_id=tokenizer.eos_token_id
            )
        
        input_length = inputs['input_ids'].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        
        print(f"\n[User]: {q}")
        print(f"[Edgy AI]: {response}")
        print("-" * 50)

if __name__ == "__main__":
    run_inference()
