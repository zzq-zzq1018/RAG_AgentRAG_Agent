import json
import time
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

# 模拟Qwen3模型调用
class MockQwen3Model:
    def __init__(self):
        self.confidence_threshold = 0.7
    
    def generate(self, prompt: str, context: List[str] = None) -> Dict[str, Any]:
        time.sleep(0.5)
        response = f"基于提供的信息：{context}，我来回答：{prompt[:50]}..."
        confidence = np.random.uniform(0.5, 1.0)
        return {
            "response": response,
            "confidence": confidence,
            "used_context": context
        }

# 模拟向量数据库
class MockVectorDB:
    def __init__(self):
        self.documents = [
            "RAG是检索增强生成，结合检索与生成能力",
            "Agent是具有自主决策能力的智能体",
            "Qwen3是阿里云通义千问3系列大模型",
            "动态RAG可根据问题调整检索策略",
            "多跳问题需要多步骤推理解决"
        ]
    
    def search(self, query: str, top_k: int = 2) -> List[str]:
        time.sleep(0.2)
        results = []
        for doc in self.documents:
            if any(keyword in doc for keyword in query.lower().split()):
                results.append(doc)
        return results[:top_k]

@dataclass
class Tool:
    name: str
    description: str
    func: callable

class AgentTools:
    def __init__(self, vector_db: MockVectorDB):
        self.vector_db = vector_db
    
    def retrieve_knowledge(self, query: str) -> List[str]:
        return self.vector_db.search(query)
    
    def decompose_question(self, question: str) -> List[str]:
        if "什么是" in question and "和" in question:
            q1, q2 = question.split("和")
            return [q1+"?", q2.replace("?", "")+"?"]
        return [question]

class RAGAgent:
    def __init__(self):
        self.model = MockQwen3Model()
        self.vector_db = MockVectorDB()
        self.tools = AgentTools(self.vector_db)
        self.memory = []
        self.max_retries = 2
        
        self.available_tools = [
            Tool("retrieve_knowledge", "检索知识库", self.tools.retrieve_knowledge),
            Tool("decompose_question", "分解复杂问题", self.tools.decompose_question)
        ]
    
    def task_classifier(self, question: str) -> str:
        if len(question.split()) > 10 or "和" in question or "如何" in question:
            return "complex"
        return "simple"
    
    def select_tools(self, task_type: str) -> List[Tool]:
        if task_type == "complex":
            return [self.available_tools[1], self.available_tools[0]]
        return [self.available_tools[0]]
    
    def dynamic_rag(self, question: str, retry_count: int = 0) -> Dict[str, Any]:
        task_type = self.task_classifier(question)
        selected_tools = self.select_tools(task_type)
        
        context = []
        tool_results = {}
        for tool in selected_tools:
            if tool.name == "decompose_question":
                subs = tool.func(question)
                tool_results["subs"] = subs
                for q in subs:
                    context.extend(self.tools.retrieve_knowledge(q))
            else:
                context.extend(tool.func(question))
        
        res = self.model.generate(question, context)
        
        if res["confidence"] < 0.7 and retry_count < 2:
            return self.dynamic_rag(question, retry_count+1)
        
        self.memory.append({
            "q": question, "type": task_type,
            "tools": [t.name for t in selected_tools],
            "conf": res["confidence"], "ans": res["response"]
        })
        
        return {
            "q": question, "type": task_type,
            "tools": [t.name for t in selected_tools],
            "ans": res["response"], "conf": res["confidence"],
            "retry": retry_count
        }

def run_experiment():
    agent = RAGAgent()
    qs = ["什么是RAG？","什么是Agent？","RAG和Agent有什么区别？",
          "如何使用Qwen3构建Agent应用？","动态RAG的作用是什么？"]
    
    results = []
    for q in qs:
        print(f"\nQ: {q}")
        s = time.time()
        r = agent.dynamic_rag(q)
        r["time"] = time.time()-s
        results.append(r)
        print(f"A: {r['ans'][:50]}...")
        print(f"置信度: {r['conf']:.2f} 工具: {r['tools']}")
    
    # 指标
    total = len(results)
    acc = sum(1 for x in results if x["conf"]>=0.7)/total
    avg_t = np.mean([x["time"] for x in results])
    multi = [x for x in results if x["type"]=="complex"]
    multi_acc = sum(1 for x in multi if x["conf"]>=0.7)/len(multi) if multi else 0
    tool_acc = sum(1 for x in results if len(x["tools"])>0)/total
    
    print(f"\n=== 评估指标 ===")
    print(f"准确率: {acc:.1%}")
    print(f"平均时间: {avg_t:.2f}s")
    print(f"多跳解决率: {multi_acc:.1%}")
    print(f"工具准确率: {tool_acc:.1%}")
    return agent, results

if __name__ == "__main__":
    agent, results = run_experiment()