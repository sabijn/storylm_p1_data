from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def chat_hf(messages, max_new_tokens=512, temperature=0.7, top_p=0.95):
    # Build the model input using the model's chat template
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (not the prompt)
    new_tokens = output_ids[0, input_ids.shape[-1]:]

    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

if __name__ == '__main__':

    # model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_id = "google/gemma-3-4b-it"

    persona_system_prompt = f"""
    Je bent een verteller van een kort verhaal (rond de 200 woorden).
    Je bent een kind tussen de 4 en 6 en je vertelt een verhaal aan klasgenoten. 
    Je publiek bestaat uit kinderen van jouw leeftijd. 
    Geef het verhaal geen titel of introductie.
    """
    n = 5  # number of generations

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",  # spreads across GPU/CPU automatically
    )

    for i in tqdm(range(n)):
        response = chat_hf(
            messages=[
                {"role": "system", "content": persona_system_prompt},
                {"role": "user", "content": "Vertel een verhaal."},
            ]
        )
        print(response)