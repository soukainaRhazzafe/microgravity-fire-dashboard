from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
SAVE_PATH = "models/smollm2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(
    SAVE_PATH,
    safe_serialization=True,
)

print(f"Model and tokenizer saved to {SAVE_PATH}")