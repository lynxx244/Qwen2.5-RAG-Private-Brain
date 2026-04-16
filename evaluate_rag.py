import json
import os
from modules.rag_engine import QwenRAGEngine

def main():
    # 初始化引擎
    engine = QwenRAGEngine()
    
    # 加载评估数据集
    with open("eval_data.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    
    evaluation_results = []
    
    print(f"📊 开始自动化评估，共 {len(test_cases)} 条测试用例...")
    
    for item in test_cases:
        query = item["question"]
        print(f"🔎 正在处理问题: {query}")
        
        # 1. 检索
        retrieved_docs = engine.retrieve(query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        
        # 2. 生成回答
        answer = engine.answer(query, context)
        
        # 3. 记录
        evaluation_results.append({
            "question": query,
            "contexts": [doc.page_content for doc in retrieved_docs],
            "answer": answer,
            "ground_truth": item["ground_truth"]
        })

    # 保存评估原始数据
    with open("eval_results_final.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    
    print("✨ 评估数据收集完成！请查看 eval_results_final.json")

if __name__ == "__main__":
    main()
