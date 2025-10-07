# 檔案名稱: chat.py
# 描述: 最終版 RAG 問答應用程式。從指定的資料庫資料夾載入知識，
#       採用穩健的後處理策略確保輸出格式穩定，並自動記錄對話。

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

# --- 模型設定 ---
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
# !!! 請務必將此路徑修改為您實際的、完整的 Gemma 模型本地路徑 !!!
GEMMA_MODEL_PATH = r"C:\Users\litsu\Desktop\gemma-3-1b" 

# --- 路徑設定 ---
# 指定資料庫檔案所在的資料夾
DATABASE_DIRECTORY = "data" 
# 自動組合檔案路徑
INDEX_FILE = os.path.join(DATABASE_DIRECTORY, "holyword.index")
CHUNKS_FILE = os.path.join(DATABASE_DIRECTORY, "holyword.json")

# 指定日誌檔案儲存的資料夾
LOG_DIRECTORY = "gemma-chat-history"
# 每次運行都創建一個新的日誌檔，以時間戳命名
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
        print(f"\n錯誤：找不到資料庫檔案 '{INDEX_FILE}' 或 '{CHUNKS_FILE}'。")
        print(f"請先成功運行 'manage_database.py' 來生成這些文件。")
        exit()

    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_PATH)
    llm_model = AutoModelForCausalLM.from_pretrained(
        GEMMA_MODEL_PATH,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    return embedding_model, index, chunks_with_metadata, tokenizer, llm_model

def retrieve_relevant_chunks(query, embedding_model, index, chunks_with_metadata, k=10):
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
        # 將每個 chunk 分句
        sentences = re.split(r'(?<=[。！？])\s*', chunk['chunk'])
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            continue

        # 計算總結與該 chunk 中所有句子的相似度
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
    穩健的生成與格式化流程：模型負責總結，Python 負責格式化。
    返回兩個值：用於顯示的完整答案，和用於記錄的純粹總結。
    """
    if not context_with_metadata:
        error_message = "無法在經文中找到相關內容來回答您的問題。"
        return error_message, error_message

    context_text = "\n\n".join([item['chunk'] for item in context_with_metadata])
    
    # 簡化的 Prompt，只要求模型做總結
    prompt_template = """<bos><start_of_turn>user
你是一位精通聖言的牧師及老師，使用者可能會向你提問、訴苦、甚至只是想請你幫他隨機挑一句聖言而已。
請你根據以下提供的聖言原文，用**一句話**用比較口語的方式來回答使用者的問題，用心去安慰或激勵那些受挫、受傷、失去鬥志的人士。
對於只想挑聖言的人而言，只回復聖言的原句、書名、頁數即可

**範例：** 
問題/訴苦：人跟神的關係是什麼? 
model: 每個人跟神都是父子關係，我們都是神的子女。

**範例：**
問題/訴苦:最近生活好辛苦
model: 如果最近生活很辛苦，可以換一些方式，就像家中的擺設，與其數個月都一成不變，不如對調一下、重新布置，那麼，想法會不同，生活會有不一樣的變化。

**範例：** 
問題/訴苦：幫我像旨意之路選一句聖言就好 
model: "不僅是人，所有萬物都渴望真愛。因此，作為萬物之靈的人類肩負著責任，不但要擁抱與愛護神創造的萬物，還要教導所有萬物如何相愛。"  平和經 1059頁
--- 原文參考 ---
{context}
---

問題：{query}<end_of_turn>
<start_of_turn>model
"""
    final_prompt = prompt_template.format(context=context_text, query=query)
    
    inputs = tokenizer(final_prompt, return_tensors="pt", max_length=4096, truncation=True).to(llm_model.device)
    outputs = llm_model.generate(**inputs, max_new_tokens=100, temperature=0.7)
    output_token_ids = outputs[0][len(inputs["input_ids"][0]):]
    
    # 這是要用於記錄的純粹總結
    model_summary_for_log = tokenizer.decode(output_token_ids, skip_special_tokens=True).strip()

    # 使用 Python 尋找最佳聖言和對應頁碼
    best_quote, best_page = find_best_quote(model_summary_for_log, context_with_metadata, embedding_model)
    
    # 動態獲取書名
    book_name = context_with_metadata[0].get('book_name', '經文')
    
    # 使用 Python 進行最終格式化
    final_formatted_answer_for_display = (
        f"model: {model_summary_for_log}\n\n"
        f"聖言: \"{best_quote}\"\n"
        f"{book_name}: {best_page}頁"
    )
    
    # 返回兩個值
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
    print("您可以開始提問了 (輸入 'exit' 或 'quit' 即可結束程式)")
    
    try:
        while True:
            user_query = input("\n[您]：")
            if user_query.lower() in ['結束服務', 'exit', 'quit']:
                break
            
            start_time = time.time()
            print("等我一下...")
            relevant_context = retrieve_relevant_chunks(user_query, embed_model, faiss_index, text_chunks)
            
            print("思考中...")
            display_answer, log_output = generate_and_format_answer(
                user_query, relevant_context, gemma_tokenizer, gemma_model, embed_model
            )
            
            print("快好了~")
            print(display_answer)

            # 建立日誌條目
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "instruction": user_query,
                "context": [item['chunk'] for item in relevant_context],
                "output": log_output  # 記錄純粹的總結
            }
            qa_history.append(log_entry)
            
            end_time = time.time()
            print(f"\n(耗時: {end_time - start_time:.2f} 秒)")

    finally:
        # 只在迴圈結束後儲存一次
        if qa_history:
            save_log(qa_history, LOG_DIRECTORY, LOG_FILE)
        else:
            print("\n對談結束，沒有新的紀錄需要儲存。")