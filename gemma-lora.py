import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel # <--- 新增 import
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import datetime
import os
import re

# --- 1. 全域設定 ---
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
BASE_MODEL_PATH = r"C:\Users\litsu\Desktop\qwen3-4b" 
LORA_ADAPTER_PATH = "gemma-lora/qwen3-4b-holyword-lora-v1" # 請確認這個路徑是正確的
DATABASE_DIRECTORY = "data"
INDEX_FILE = os.path.join(DATABASE_DIRECTORY, "holyword.index")
CHUNKS_FILE = os.path.join(DATABASE_DIRECTORY, "holyword.json")
LOG_DIRECTORY = "gemma-chat-history"
LOG_FILE = os.path.join(LOG_DIRECTORY, f"qa_log_lora_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

# --- 2. 函數定義 ---

def load_models_and_data():
    """
    載入模型、RAG 資料和 tokenizer。
    """
    print("==================================================")
    print("       正在準備中...")
    print("==================================================")
    
    # 載入 RAG 部分 (不變)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks_with_metadata = json.load(f)

    # 載入量化後的基礎模型
    print(f"正在從 '{BASE_MODEL_PATH}' 載入量化基礎模型...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True, # 處理記憶體不足
        bnb_4bit_quant_type='nf4', # 通常 nf4 表現不錯
        # bnb_4bit_use_double_quant=True, # 可能增加記憶體使用，但精確度更高
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=quantization_config,
        device_map={"": 0}, # 強制載入到 GPU 0
        torch_dtype=torch.bfloat16,
        trust_remote_code=True, # Qwen 模型需要
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

    # 合併 LoRA 適配器
    print(f"正在從 '{LORA_ADAPTER_PATH}' 合併 LoRA 適配器...")
    model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
    
    # --- 優化：設定模型為評估模式 ---
    model.eval()
    print("模型載入與優化完成！")
    
    return embedding_model, index, chunks_with_metadata, tokenizer, model

def retrieve_relevant_chunks(query, embedding_model, index, chunks_with_metadata, k=3):
    """檢索函數，返回包含元數據的完整物件列表。"""
    query_vector = embedding_model.encode([query])
    query_vector_np = np.array(query_vector).astype('float32')
    distances, indices = index.search(query_vector_np, k)
    return [chunks_with_metadata[i] for i in indices[0]]

def find_best_quote(summary, context_chunks, embedding_model):
    """從原文中找出與模型總結最相似的句子，並返回該句子和其來源頁碼。"""
    best_quote = "（無法從原文中自動摘錄聖言）"
    best_page = "N/A"
    highest_similarity = -1
    summary_embedding = embedding_model.encode([summary])
    for chunk in context_chunks:
        sentences = re.split(r'(?<=[。！？])\s*', chunk['chunk'])
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences: continue
        sentence_embeddings = embedding_model.encode(sentences)
        similarities = cosine_similarity(summary_embedding, sentence_embeddings)[0]
        local_best_index = np.argmax(similarities)
        local_highest_similarity = similarities[local_best_index]
        if local_highest_similarity > highest_similarity:
            highest_similarity = local_highest_similarity
            best_quote = sentences[local_best_index]
            best_page = chunk['page']
    return best_quote, best_page

def extract_first_sentence(text):
    """從文字中提取第一句話。"""
    # 使用正規表達式分割句子，以句號、驚嘆號、問號為界
    sentences = re.split(r'[。！？]', text)
    # 找到第一個非空的句子
    for sentence in sentences:
        clean_sentence = sentence.strip()
        if clean_sentence:
            return clean_sentence
    # 如果沒有找到，返回原文字
    return text.strip()

def generate_and_format_answer(query, context_with_metadata, tokenizer, llm_model, embedding_model):
    """
    此函數現在會返回兩個值：
    1. 用於顯示的完整格式化答案。
    2. 只包含模型總結的純文字，用於記錄。
    """
    if not context_with_metadata:
        error_message = "無法在經文中找到相關內容來回答您的問題。"
        return error_message, error_message

    context_text = "\n\n".join([item['chunk'] for item in context_with_metadata])
    
    # --- 優化提示：更強調「一句話」 ---
    prompt_template = """你是一位精通聖言的牧師及老師，根據以下提供的聖言原文，用**一句話**、口語化的方式回答問題。

**範例：** 
問題/訴苦：人跟神的關係是什麼? 
model: 每個人跟神都是父子關係，我們都是神的子女。

--- 原文參考 ---
{context}

問題：{query}

model："""
    final_prompt = prompt_template.format(context=context_text, query=query)
    
    inputs = tokenizer(final_prompt, return_tensors="pt", max_length=4096, truncation=True).to(llm_model.device)
    outputs = llm_model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id, do_sample=False) # 加入 do_sample=False 以求更穩定輸出
    output_token_ids = outputs[0][len(inputs["input_ids"][0]):]
    
    # 解碼模型輸出
    raw_model_output = tokenizer.decode(output_token_ids, skip_special_tokens=True).strip()
    
    # --- 優化：提取第一句話作為核心回答 ---
    concise_summary = extract_first_sentence(raw_model_output)
    
    best_quote, best_page = find_best_quote(concise_summary, context_with_metadata, embedding_model)
    book_name = context_with_metadata[0].get('book_name', '經文')
    
    # 完整格式化答案（用於顯示）
    final_formatted_answer_for_display = (
        f"model: {concise_summary}\n\n"  # 這裡改用 concise_summary
        f"聖言: \"{best_quote}\"\n"
        f"{book_name}: {best_page}頁"
    )
    
    # 純粹總結（用於記錄）
    model_summary_for_log = concise_summary
    
    return final_formatted_answer_for_display, model_summary_for_log

def save_log(log_data, directory, filename):
    """將問答紀錄儲存到指定資料夾和檔案中。"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    full_path = os.path.join(directory, os.path.basename(filename))
    try:
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        print(f"\n對談結束，已將 {len(log_data)} 筆紀錄儲存至 {full_path}")
    except Exception as e:
        print(f"\n儲存紀錄時發生錯誤: {e}")

# --- 主程式執行迴圈 ---
if __name__ == "__main__":
    embed_model, faiss_index, text_chunks, gemma_tokenizer, gemma_model = load_models_and_data()
    
    qa_history = [] 

    print("\n==================================================")
    print("      歡迎來到AI版的《旨意之路》")
    print("==================================================")
    print("您可以開始提問了 (輸入 '結束服務', 'exit' 或 'quit' 即可結束程式)")
    
    try:
        while True:
            user_query = input("\n[您]：")
            if user_query.lower() in ['結束服務', 'exit', 'quit']:
                break
            
            start_time = time.time()
            print("等我一下...")
            relevant_context = retrieve_relevant_chunks(user_query, embed_model, faiss_index, text_chunks)
            
            print("思考中...")
            final_answer_for_display, model_answer_for_log = generate_and_format_answer(
                user_query, relevant_context, gemma_tokenizer, gemma_model, embed_model
            )
            
            print("快好了~")
            print(final_answer_for_display)

            # 純粹總結已由 extract_first_sentence 確保簡潔
            pure_summary = model_answer_for_log # 這時 pure_summary 就是第一句話
            
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "instruction": user_query,
                "context": [item['chunk'] for item in relevant_context],
                "output": pure_summary
            }
            qa_history.append(log_entry)
            
            end_time = time.time()
            print(f"\n(耗時: {end_time - start_time:.2f} 秒)")

    finally:
        if qa_history:
            save_log(qa_history, LOG_DIRECTORY, LOG_FILE)
        else:
            print("\n對談結束，沒有新的紀錄需要儲存。")
