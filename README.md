# Qwen2.5-RAG-Private-Brain
# 🤖 Qwen-RAG-Brain: 基于 Qwen2.5 的进阶 RAG 智能知识库

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Qwen](https://img.shields.io/badge/Qwen2.5--7B-blue?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-⚡-green?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-yellow?style=for-the-badge)

本项目是一个基于 **Qwen2.5-7B** 大模型构建的本地工业级 RAG（检索增强生成）系统。相比于传统的初级 RAG，本项目集成了 **BGE-Reranker 精排模型**、**多轮对话状态机（Session State）**以及**复杂 PDF 文档解析**，有效解决了垂直领域问答中的模型幻觉、长文本召回不准与多轮意图遗忘等痛点问题。

---

## 🌟 核心特性 (Key Features)

* **⚡ 双层检索架构 (Two-Stage Retrieval)**：采用 FAISS 与 BGE-Small 进行初筛粗排，并引入 `BGE-Reranker-v2-m3` 进行深度语义重排序，确保大模型的上下文具有最高相关性和忠实度。
* **🧠 智能对话记忆 (Context-aware Memory)**：基于 Streamlit Session State 构建多轮对话记忆，支持跨轮指代消解（如连问“什么是 QKV”、“它怎么计算”），实现连贯的私人助理体验。
* **📄 深度文档解析 (Robust Parsing)**：针对技术 PDF 和 Markdown，利用 `RecursiveCharacterTextSplitter` 进行递归分块，并设置科学的 Token 语义重叠（Overlap），完美保护长难句与复杂数学公式。
* **🛠️ 强健的工程实践 (Engineering Excellence)**：应对底层依赖地狱，通过 **Monkey Patch (猴子补丁)** 动态修复 `transformers` 与 `FlagEmbedding` 之间的 API 冲突（如 Tokenizer 属性缺失）。

---

## 🛠️ 技术栈 (Tech Stack)

* **大型语言模型 (LLM)**: `Qwen/Qwen2.5-7B-Instruct`
* **词嵌入模型 (Embedding)**: `BAAI/bge-small-zh-v1.5`
* **重排序模型 (Reranker)**: `BAAI/bge-reranker-v2-m3`
* **向量数据库 (Vector DB)**: `FAISS`
* **应用框架**: `LangChain`, `Streamlit`
* **部署环境**: `AutoDL` (推荐 24GB 显存，如 RTX 3090/4090)

---

## 🚀 快速启动 (Quick Start)

### 1. 环境准备
克隆本项目并安装依赖：
```bash
git clone [https://github.com/你的用户名/Qwen2.5-RAG-Brain.git](https://github.com/你的用户名/Qwen2.5-RAG-Brain.git)
cd Qwen2.5-RAG-Brain
pip install -r requirements.txt
```

### 2. 构建本地知识库
将需要解析的 PDF 或 Markdown 文档放入 `data/` 目录中，然后执行向量化脚本：
```bash
python modules/create_vector.py
```
*(注：生成的索引文件会保存在 `vector_store/` 目录下，该目录已加入 `.gitignore`。)*

### 3. 启动交互式 Web 界面
```bash
streamlit run web_app.py --server.port 6006
```

---

## 📊 运行效果展示
<img width="1906" height="883" alt="68a6c15788df321480c3ad5b382c5681" src="https://github.com/user-attachments/assets/85ae5166-cd43-4159-901c-682ed41f4a8a" />

![Uploading ec7d6927f0b8ff26d5086dee42613490.png…]()




---

## 📝 开发者札记 (Troubleshooting)

在项目构建过程中，遇到了 `XLMRobertaTokenizer` 丢失 `prepare_for_model` 属性导致 Reranker 崩溃的底层依赖冲突。本项目在主程序中采用了 Monkey Patch（猴子补丁）策略实现版本兼容，保障了系统的稳定运行。

```python
import transformers.utils.import_utils as import_utils
from transformers.models.xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer

# 修复 FlagEmbedding 与新版 transformers 的 API 不兼容问题
if not hasattr(import_utils, "is_torch_fx_available"):
    import_utils.is_torch_fx_available = lambda: False 
if not hasattr(XLMRobertaTokenizer, "prepare_for_model"):
    XLMRobertaTokenizer.prepare_for_model = XLMRobertaTokenizer._prepare_for_model
```

---
**Author**: [lynxx244]

**License**: MIT
