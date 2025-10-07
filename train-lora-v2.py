import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset

# --- 1. 全域設定 ---
BASE_MODEL_PATH = r"C:\Users\litsu\Desktop\gemma-3-1b"
OLD_LORA_PATH = "gemma-lora/gemma3-holyword-lora-v1"  # 舊 LoRA
NEW_DATASET_FILE = "gemma-chat-history/QA-data2.json"
OUTPUT_LORA_ADAPTER_PATH = "gemma-lora/gemma3-holyword-lora-v2"

# --- 2. LoRA 設定（與舊 LoRA 相同） ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- 3. 載入基底模型與舊 LoRA ---
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"載入基底模型 '{BASE_MODEL_PATH}' ...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto"
)
base_model = prepare_model_for_kbit_training(base_model)

print(f"載入舊 LoRA adapter '{OLD_LORA_PATH}' ...")
model = PeftModel.from_pretrained(base_model, OLD_LORA_PATH)
model.print_trainable_parameters()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- 4. 載入新資料集並 Tokenize ---
print(f"載入新資料集 '{NEW_DATASET_FILE}' ...")
dataset = load_dataset("json", data_files=NEW_DATASET_FILE, split="train")

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

# --- 5. 訓練設定 ---
training_args = TrainingArguments(
    output_dir=OUTPUT_LORA_ADAPTER_PATH,   # 只是儲存 LoRA adapter
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=5,
    logging_steps=10,
    save_strategy="no",     # ⚡ 不存 checkpoint
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

# --- 6. 開始增量訓練 ---
print("開始增量 LoRA 訓練 ...")
trainer.train()
print("訓練完成。")

# --- 7. 儲存新的 LoRA adapter ---
print(f"儲存新的 LoRA adapter 到 '{OUTPUT_LORA_ADAPTER_PATH}' ...")
model.save_pretrained(OUTPUT_LORA_ADAPTER_PATH)
tokenizer.save_pretrained(OUTPUT_LORA_ADAPTER_PATH)
print("完成。")
