import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# --- 1. 全域設定 ---
BASE_MODEL_PATH = r"C:\Users\litsu\Desktop\qwen3-4b"
DATASET_FILE = "gemma-chat-history/QA-data1.json"
OUTPUT_LORA_ADAPTER_PATH = "gemma-lora/qwen3-4b-holyword-lora-v1"

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- 3. 載入基底模型與 Tokenizer ---
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"載入基底模型 '{BASE_MODEL_PATH}' ...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- 4. 應用 LoRA ---
print("應用 LoRA adapter ...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- 5. 載入資料集並 Tokenize ---
print(f"載入資料集 '{DATASET_FILE}' ...")
dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

def tokenize_example(example):
    prompt_string = (
        f"<bos><start_of_turn>user\n{example['instruction']}<end_of_turn>\n"
        f"<start_of_turn>model\n{example['output']}<end_of_turn><eos>"
    )
    tokenized = tokenizer(
        prompt_string,
        truncation=True,
        max_length=1024,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Tokenizing 資料集 ...")
tokenized_dataset = dataset.map(tokenize_example)
print("Tokenization 完成。")

# --- 6. 訓練設定（不存 checkpoint） ---
training_args = TrainingArguments(
    output_dir=OUTPUT_LORA_ADAPTER_PATH,   # 只是儲存 LoRA adapter
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="no",   # 不生成 checkpoint
    fp16=True,
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --- 7. 開始訓練 ---
print("開始 LoRA 訓練 ...")
trainer.train()
print("訓練完成。")

# --- 8. 儲存 LoRA adapter 及 tokenizer ---
print(f"儲存 LoRA adapter 到 '{OUTPUT_LORA_ADAPTER_PATH}' ...")
model.save_pretrained(OUTPUT_LORA_ADAPTER_PATH)
tokenizer.save_pretrained(OUTPUT_LORA_ADAPTER_PATH)
print("完成。")

