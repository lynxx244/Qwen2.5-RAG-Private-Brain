import os
# 1. 严格要求：环境初始化必须放在最顶层
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"

import torch
import logging
import transformers.utils.import_utils as import_utils
from transformers.models.xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer

# 2. 严格要求：独立模块必须自包含补丁逻辑，确保鲁棒性
if not hasattr(import_utils, "is_torch_fx_available"):
    import_utils.is_torch_fx_available = lambda: False 
if not hasattr(XLMRobertaTokenizer, "prepare_for_model"):
    XLMRobertaTokenizer.prepare_for_model = XLMRobertaTokenizer._prepare_for_model

# 3. 接下来才是从第 15 行开始的资源导入...
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from FlagEmbedding import FlagReranker

# 配置日志记录，这是可观测性的第一步
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QwenRAGEngine:
    def __init__(self, model_path="Qwen/Qwen2.5-7B-Instruct", vector_db_path="./vector_store/index"):
        """
        初始化 RAG 引擎：加载嵌入、向量库、精排模型和大模型
        """
        logger.info("正在初始化 RAG 引擎资源...")
        
        # 1. 加载 Embedding
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={'device': 'cpu'}
        )
        
        # 2. 加载向量库
        self.vector_db = FAISS.load_local(
            vector_db_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # 3. 加载 Reranker (精排)
        self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        
        # 4. 加载 Qwen2.5
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        logger.info("✅ 所有模型资源加载完毕。")

    def retrieve(self, query, k_initial=10, k_final=3):
        """
        执行双重检索：向量初筛 + Reranker 精排
        """
        initial_docs = self.vector_db.similarity_search(query, k=k_initial)
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.reranker.compute_score(pairs)
        
        # 按照精排分数排序并截取
        scored_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:k_final]]

    def answer(self, query, context):
        """
        根据上下文生成回答
        """
        prompt = f"请根据以下背景资料回答问题。如果资料中没有相关信息，请诚实说明。\n\n【背景资料】\n{context}\n\n【问题】\n{query}\n\n回答："
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.split("assistant\n")[-1].strip()
