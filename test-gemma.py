import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
GEMMA_MODEL_PATH = r"C:\Users\litsu\Desktop\qwen3-4b" # 您的本地路徑
DATABASE_DIRECTORY = "data"
INDEX_FILE = os.path.join(DATABASE_DIRECTORY, "holyword.index")
CHUNKS_FILE = os.path.join(DATABASE_DIRECTORY, "holyword.json")
LOG_DIRECTORY = "qwen-chat-history" # Changed to reflect Qwen model
LOG_FILE = os.path.join(LOG_DIRECTORY, f"qa_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

# --- 2. 函數定義 ---

def load_models_and_data():
    """載入所有必要的模型和資料檔案。"""
    print("==================================================")
    print("      《平和經》智慧導師 正在準備中...")
    print("==================================================")
    print("首次啟動時模型載入較慢，請耐心等候...")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    try:
        index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            chunks_with_metadata = json.load(f)
    except FileNotFoundError:
        print(f"\n錯誤：找不到 {INDEX_FILE} 或 {CHUNKS_FILE}。")
        print("請先成功運行 'create_database.py' 來生成這些文件。")
        exit()

    # --- 修正量化配置 ---
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        # 移除 llm_int8_enable_fp32_cpu_offload=True
        # 並使用更穩定的 device_map
        bnb_4bit_quant_type='nf4', # 可選，通常 nf4 表現不錯
        # bnb_4bit_use_double_quant=True, # 可選，但可能增加記憶體使用
    )

    tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_PATH)
    # 使用 device_map={"": 0} 將整個模型強制載入到 GPU 0
    # 並加入 trust_remote_code=True 以支援 Qwen
    llm_model = AutoModelForCausalLM.from_pretrained(
        GEMMA_MODEL_PATH,
        quantization_config=quantization_config,
        device_map={"": 0}, # 強制載入到 GPU 0，避免 meta tensor 問題
        torch_dtype=torch.bfloat16, # 載入權重的資料類型
        trust_remote_code=True, # Qwen 模型需要此參數
    )

    return embedding_model, index, chunks_with_metadata, tokenizer, llm_model

# ... (其餘函數 retrieve_relevant_chunks, find_best_quote, generate_and_format_answer, save_log 保持不變)
# 請將您之前版本的這些函數複製到這裡
# 但請注意 generate_and_format_answer 中的 prompt_template 需要配合 Qwen 格式
# 以及可能需要調整 pad_token_id

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

def generate_and_format_answer(query, context_with_metadata, tokenizer, llm_model, embedding_model):
    """
    此函數現在會返回兩個值：
    1. 用於顯示的完整格式化答案。
    2. 只包含模型總結的純文字，用於記錄。
    """
    if not context_with_metadata:
        # 如果找不到上下文，兩個返回值都設為同樣的錯誤訊息
        error_message = "無法在經文中找到相關內容來回答您的問題。"
        return error_message, error_message

    context_text = "\n\n".join([item['chunk'] for item in context_with_metadata])
    
    # --- 修正 Qwen 提示模板 ---
    prompt_template = """<|system|>
你是一位精通聖言的牧師及老師，根據以下提供的聖言原文，用**一句話**用比較口語的方式來回答使用者的問題。

**範例：** 
問題/訴苦：人跟神的關係是什麼? 
model: 每個人跟神都是父子關係，我們都是神的子女。
</s>
</s>
<|user|>
--- 原文參考 ---
{context}
---

問題：{query}</s>
<|assistant|>
"""
    final_prompt = prompt_template.format(context=context_text, query=query)
    
    inputs = tokenizer(final_prompt, return_tensors="pt", max_length=4096, truncation=True).to(llm_model.device)
    # --- 修正 generate 參數 ---
    # 移除 pad_token_id，因為 Qwen 通常不需要，且可能導致問題
    # 或者，如果需要，可以使用 tokenizer.pad_token_id
    outputs = llm_model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    output_token_ids = outputs[0][len(inputs["input_ids"][0]):]
    
    # *** 1. 這是要用於記錄的純粹總結 ***
    model_summary_for_log = tokenizer.decode(output_token_ids, skip_special_tokens=True).strip()

    best_quote, best_page = find_best_quote(model_summary_for_log, context_with_metadata, embedding_model)
    book_name = context_with_metadata[0].get('book_name', '經文')
    
    # *** 2. 這是要用於顯示的完整格式 ***
    final_formatted_answer_for_display = (
        f"model: {model_summary_for_log}\n\n"
        f"聖言: \"{best_quote}\"\n"
        f"{book_name}: {best_page}頁"
    )
    
    # *** 3. 一次返回兩個值 ***
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
            # *** 核心修改：接收兩個返回值 ***
            final_answer_for_display, model_answer_for_log = generate_and_format_answer(
                user_query, relevant_context, gemma_tokenizer, gemma_model, embed_model
            )
            
            print("快好了~")
            # 使用格式化的版本來顯示
            print(final_answer_for_display)

            # *** 核心修改：用純粹的總結來記錄 ***
            # 我們需要把 model: 這個前綴也去掉，只保留純粹的句子
            pure_summary = model_answer_for_log.replace("model:", "").strip()
            
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "instruction": user_query,
                "context": [item['chunk'] for item in relevant_context],
                "output": pure_summary  # <-- 只記錄模型的純粹總結
            }
            qa_history.append(log_entry)
            
            end_time = time.time()
            print(f"\n(耗時: {end_time - start_time:.2f} 秒)")

    finally:
        # 這個區塊只會在 while 迴圈結束後執行一次
        if qa_history:
            save_log(qa_history, LOG_DIRECTORY, LOG_FILE)
        else:
            print("\n對談結束，沒有新的紀錄需要儲存。")
