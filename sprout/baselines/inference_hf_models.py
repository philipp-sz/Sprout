import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_text(prompt, max_new_tokens=50, temperature=0.8, top_k=None, top_p=0.95):
    model_name = "gpt2"
    
    print(f" Loading {model_name}... ")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("Generating...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(output_ids[0])
    return generated_text

if __name__ == "__main__":
    prompt_input = "Newton's first law of motion states that an object at rest remains at rest, and an object in motion stays in motion at constant velocity, unless acted upon by"
    
    result = generate_text(prompt_input)

    print("\n--- Model Output ---")
    print(result)