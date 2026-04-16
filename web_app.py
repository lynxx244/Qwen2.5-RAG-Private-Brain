# --- 1. 强制补丁：修复 transformers 版本不兼容问题 (放在最开头) ---
import transformers.utils.import_utils as import_utils
if not hasattr(import_utils, "is_torch_fx_available"):
    import_utils.is_torch_fx_available = lambda: False 
from modules.rag_engine import QwenRAGEngine
import streamlit as st
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from FlagEmbedding import FlagReranker

# --- 2. 页面配置 ---
st.set_page_config(page_title="Qwen2.5 进阶 RAG 助手", page_icon="🤖", layout="wide")

# --- 3. 资源初始化 (使用缓存避免重复加载) ---
@st.cache_resource
def init_all():
    # A. Embedding 模型
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5", 
        model_kwargs={'device': 'cpu'}
    )
    # B. 加载向量库
    vector_db = FAISS.load_local(
        "./vector_store/index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    # C. Reranker 精排模型
    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True) 
    # D. Qwen2.5 大模型
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    return vector_db, reranker, tokenizer, model

with st.spinner("🚀 系统正在初始化，加载模型中..."):
    vector_db, reranker, tokenizer, model = init_all()

# --- 4. 核心功能函数 ---

def get_advanced_context(query, k_initial=10, k_final=3):
    """双重检索：初筛 + 精排"""
    # 初筛
    initial_docs = vector_db.similarity_search(query, k=k_initial)
    # 精排
    pairs = [[query, doc.page_content] for doc in initial_docs]
    scores = reranker.compute_score(pairs)
    # 排序并截取
    scored_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:k_final]]

def build_chat_prompt(history, context, current_query):
    """构造带记忆和背景的 Prompt"""
    history_str = ""
    # 只取最近 3 轮对话，防止上下文过长
    for msg in history[-3:]:
        role = "用户" if msg["role"] == "user" else "助手"
        history_str += f"{role}: {msg['content']}\n"
    
    prompt = f"""你是一个专业的 AI 助手。请根据提供的背景资料和对话历史，专业且简洁地回答用户问题。
如果背景资料中没有相关信息，请诚实说明。

【背景资料】
{context}

【对话历史】
{history_str}

【当前问题】
{current_query}

回答："""
    return prompt

# --- 5. 对话界面逻辑 ---

# 初始化对话历史笔记本
if "messages" not in st.session_state:
    st.session_state.messages = []

# 侧边栏
with st.sidebar:
    st.header("📊 知识库状态")
    st.success("向量数据库：已就绪")
    st.info(f"精排模型：BGE-Reranker-v2-M3")
    if st.button("🧹 清除聊天记录"):
        st.session_state.messages = []
        st.rerun()

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 处理当前用户输入
if prompt := st.chat_input("问我关于文档的问题..."):
    # 1. 立即展示并记录用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 生成助手回答
    with st.chat_message("assistant"):
        # A. 检索背景资料
        with st.status("🔍 正在深度搜索知识库...", expanded=False) as status:
            top_docs = get_advanced_context(prompt)
            context = "\n".join([f"[来自 {d.metadata.get('source', '未知')}] {d.page_content}" for d in top_docs])
            for i, d in enumerate(top_docs):
                st.write(f"**证据 {i+1}:** {d.page_content[:150]}...")
            status.update(label="✅ 搜索完成，正在组织语言...", state="complete")

        # B. 构造 Prompt 并调用大模型
        full_prompt = build_chat_prompt(st.session_state.messages[:-1], context, prompt)
        
        with st.spinner("🤖 Qwen 正在思考..."):
            messages = [{"role": "user", "content": full_prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            generated_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            answer = response.split("assistant\n")[-1]

        # C. 展示并记录结果
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
