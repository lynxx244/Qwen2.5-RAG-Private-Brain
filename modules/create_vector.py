import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from load_data import process_document

def create_multi_vector_db():
    data_dir = "./data"
    all_chunks = []
    
    # 遍历 data 文件夹下所有的 .md 和 .pdf 文件
    for file_name in os.listdir(data_dir):
        if file_name.endswith(('.md', '.pdf')):
            print(f"🔍 发现新文档: {file_name}")
            chunks = process_document(file_name)
            if chunks:
                all_chunks.extend(chunks)
    
    if not all_chunks:
        print("❌ 错误：没有找到可处理的文档！")
        return

    print(f"⏳ 正在对 {len(all_chunks)} 个片段进行向量化...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'} # 维持之前的 CPU 方案最稳
    )
    
    db = FAISS.from_documents(all_chunks, embeddings)
    db.save_local("./vector_store/index")
    print(f"✅ 全量知识库已保存！共包含 {len(all_chunks)} 个片段。")

if __name__ == "__main__":
    create_multi_vector_db()