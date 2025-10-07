# 檔案名稱: manage_database.py
# 描述: 一个可增量更新的 RAG 数据库管理脚本。
#       它会自动检测 source_texts 中新增的 txt 档案，并只处理新档案，
#       然后将新数据追加到现有的 FAISS 索引和 JSON 檔案中。

import re
import json
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import hashlib

# --- 1. 全域設定 ---
SOURCE_DIRECTORY = "orgin-file"
DATABASE_DIRECTORY = "data"
INDEX_FILE = os.path.join(DATABASE_DIRECTORY, "holyword.index")
CHUNKS_FILE = os.path.join(DATABASE_DIRECTORY, "holyword.json")
# 新增：用於記錄已处理檔案及其状态的檔案
PROCESSED_LOG_FILE = os.path.join(DATABASE_DIRECTORY, "processed_log.json")
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
SIMILARITY_THRESHOLD = 0.6

# --- 2. 函數定義 (大部分与之前相同) ---
# ... clean_and_split_sentences 和 merge_sentences_semantically 函數保持不变 ...
def clean_and_split_sentences(raw_text, book_name):
    # (此處省略以保持簡潔)
    PAGE_RE = re.compile(r'---\s*Page\s*(\d+)\s*---', re.IGNORECASE)
    last_seen_page, sentence_start_page, in_quote, buf, sentences_with_pages = '1', '1', False, [], []
    i, n = 0, len(raw_text)
    while i < n:
        m = PAGE_RE.match(raw_text, i)
        if m: last_seen_page, i = m.group(1), m.end(); continue
        ch = raw_text[i]
        if ch == '「': in_quote = True
        elif ch == '」': in_quote = False
        if not buf and not ch.isspace(): sentence_start_page = last_seen_page
        if ch.isspace():
            if buf and buf[-1] != ' ': buf.append(' ')
        else: buf.append(ch)
        if not in_quote and ch in "。！？；!?.":
            sentence = ''.join(buf).strip()
            if sentence: sentences_with_pages.append({"book_name": book_name, "page": sentence_start_page, "sentence": re.sub(r'\s+', '', sentence)})
            buf, sentence_start_page = [], last_seen_page
        i += 1
    if buf:
        sentence = ''.join(buf).strip()
        if sentence: sentences_with_pages.append({"book_name": book_name, "page": sentence_start_page, "sentence": re.sub(r'\s+', '', sentence)})
    return sentences_with_pages

def merge_sentences_semantically(sentences_with_pages, model, threshold):
    # (此處省略以保持簡潔)
    if not sentences_with_pages: return []
    sentences_only = [item["sentence"] for item in sentences_with_pages]
    embeddings = model.encode(sentences_only, show_progress_bar=True, batch_size=16)
    final_chunks, current_chunk_sentences, current_chunk_start_page, current_chunk_book_name = [], [], None, None
    for i in range(len(sentences_with_pages)):
        if not current_chunk_sentences:
            current_chunk_start_page = sentences_with_pages[i]["page"]
            current_chunk_book_name = sentences_with_pages[i]["book_name"]
        current_chunk_sentences.append(sentences_with_pages[i]["sentence"])
        is_last_sentence = (i == len(sentences_with_pages) - 1)
        is_below_threshold = not is_last_sentence and cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0] < threshold
        if is_last_sentence or is_below_threshold:
            final_chunks.append({"book_name": current_chunk_book_name, "page": current_chunk_start_page, "chunk": "".join(current_chunk_sentences)})
            current_chunk_sentences = []
    return [c for c in final_chunks if len(c['chunk']) > 30]

# --- 3. 新增的辅助函數 ---

def get_file_hash(filepath):
    """计算檔案的 SHA256 哈希值，用于检测檔案是否被修改。"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def load_processed_log():
    """載入已处理檔案的日志。"""
    if os.path.exists(PROCESSED_LOG_FILE):
        with open(PROCESSED_LOG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_processed_log(log_data):
    """保存已处理檔案的日志。"""
    with open(PROCESSED_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

# --- 4. 主执行流程 (已重构) ---
if __name__ == "__main__":
    start_time = time.time()
    
    # 确保资料夹存在
    if not os.path.exists(SOURCE_DIRECTORY):
        os.makedirs(SOURCE_DIRECTORY)
        print(f"已创建来源资料夹 '{SOURCE_DIRECTORY}'。请将您的 .txt 档案放入其中。")
    if not os.path.exists(DATABASE_DIRECTORY):
        os.makedirs(DATABASE_DIRECTORY)

    # 載入日志和现有数据
    processed_log = load_processed_log()
    
    # 找出需要处理的新檔案
    new_files_to_process = []
    print(f"--- 正在扫描 '{SOURCE_DIRECTORY}' 中的檔案 ---")
    for filename in os.listdir(SOURCE_DIRECTORY):
        if filename.endswith(".txt"):
            file_path = os.path.join(SOURCE_DIRECTORY, filename)
            current_hash = get_file_hash(file_path)
            # 如果檔案是新的，或者檔案内容已更新，则需要重新处理
            if filename not in processed_log or processed_log[filename] != current_hash:
                new_files_to_process.append(filename)
                print(f"  - 检测到新檔案或已更新: {filename}")

    if not new_files_to_process:
        print("--- 没有需要更新的檔案。数据库已是最新状态。 ---")
        exit()

    print(f"\n--- 准备处理 {len(new_files_to_process)} 个新檔案 ---")
    
    # 载入模型 (只有在需要处理新檔案时才载入)
    print("正在载入嵌入模型 (BAAI/bge-m3)...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    newly_generated_chunks = []
    for filename in new_files_to_process:
        book_name = os.path.splitext(filename)[0]
        file_path = os.path.join(SOURCE_DIRECTORY, filename)
        print(f"\n正在处理书本: '{book_name}'")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        
        sentences = clean_and_split_sentences(raw_text, book_name)
        chunks = merge_sentences_semantically(sentences, embedding_model, SIMILARITY_THRESHOLD)
        newly_generated_chunks.extend(chunks)
        # 更新日志
        processed_log[filename] = get_file_hash(file_path)

    print(f"\n--- 所有新檔案处理完毕，共生成 {len(newly_generated_chunks)} 个新的语意区块 ---")

    # 向量化新区块
    if newly_generated_chunks:
        print("正在将新的区块向量化...")
        new_embeddings = embedding_model.encode([c['chunk'] for c in newly_generated_chunks], show_progress_bar=True)
        new_embeddings = np.array(new_embeddings).astype('float32')

        # 加载或初始化数据库
        if os.path.exists(INDEX_FILE):
            print("正在将新数据追加到现有数据库...")
            # 载入旧的索引和 JSON
            index = faiss.read_index(INDEX_FILE)
            with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
                all_chunks = json.load(f)
            
            # 追加新数据
            index.add(new_embeddings)
            all_chunks.extend(newly_generated_chunks)
        else:
            print("正在创建新的数据库...")
            # 创建新的索引和 JSON
            dimension = new_embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(new_embeddings)
            all_chunks = newly_generated_chunks
        
        # 保存更新后的数据库和日志
        faiss.write_index(index, INDEX_FILE)
        with open(CHUNKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        save_processed_log(processed_log)
        
        print(f"✅ 数据库更新成功！现在总共有 {index.ntotal} 个区块。")
    
    end_time = time.time()
    print(f"本次更新耗时: {end_time - start_time:.2f} 秒")