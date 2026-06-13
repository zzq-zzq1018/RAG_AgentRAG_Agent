# Dynamic RAG Agent with Tool Use

基于 Qwen3 的 **动态检索增强生成（Dynamic RAG）** 与 **智能体工具调用** 的模拟实现。  
项目演示了如何根据问题复杂度自动选择检索策略、分解多跳问题，并利用置信度评估实现自适应重试。

## 核心特性

- 🔍 **动态 RAG**：根据问题类型（简单/复杂）自动切换检索深度  
- 🛠️ **工具使用**：内置知识库检索、问题分解两种工具，Agent 自主决定调用顺序  
- 🔁 **置信度重试**：生成结果置信度低于阈值时自动重试，最多 2 次  
- 🧠 **模拟组件**：包含 Mock Qwen3 模型和 Mock 向量数据库，方便快速验证逻辑  
- 📊 **实验评估**：自动计算准确率、平均耗时、多跳问题解决率等指标

## 项目结构
.
├── README.md # 本文件
└── dynamic_rag_agent.py # 主程序（包含所有类与运行示例）

text

### 核心类说明

| 类名             | 职责                                                         |
| ---------------- | ------------------------------------------------------------ |
| `MockQwen3Model` | 模拟大语言模型，生成带置信度的响应，支持上下文注入           |
| `MockVectorDB`   | 模拟向量数据库，基于关键词匹配返回相关文档                   |
| `AgentTools`     | 定义 Agent 可用的工具：知识检索、问题分解                    |
| `RAGAgent`       | 智能体核心：任务分类、工具选择、动态 RAG 执行、重试机制与记忆存储 |
| `Tool`           | 工具的数据类，包含名称、描述和可调用函数                     |

## 安装与运行

### 1. 环境要求

- Python 3.8+
- 仅依赖标准库：`json`, `time`, `typing`, `dataclasses`, `numpy`

### 2. 安装依赖

```bash
pip install numpy
本项目未使用深度学习框架，所有模型均为模拟实现，无需 GPU 或额外模型文件。

3. 运行实验
bash
python dynamic_rag_agent.py
脚本会自动执行预定义的 5 个测试问题，并输出每个问题的回答片段、置信度、所用工具，最后显示整体评估指标。

配置与定制
替换为真实模型与数据库
Qwen3 模型：修改 MockQwen3Model.generate() 方法，接入阿里云 DashScope SDK 或本地推理接口。

向量数据库：替换 MockVectorDB.search() 为真实向量库（如 Chroma、FAISS、Milvus）的相似度检索。

自定义工具：在 AgentTools 中添加新方法，并在 RAGAgent.available_tools 中注册即可。

调整参数
RAGAgent.max_retries：最大重试次数（默认 2）

MockQwen3Model.confidence_threshold：置信度阈值（默认 0.7）

MockVectorDB.search(top_k)：检索返回的文档数量

运行示例输出
text
Q: 什么是RAG？
A: 基于提供的信息：['RAG是检索增强生成，结合检索与生成能力']，我来回答：什么是RAG？...
置信度: 0.85 工具: ['retrieve_knowledge']

Q: RAG和Agent有什么区别？
A: 基于提供的信息：['RAG是检索增强生成，结合检索与生成能力', 'Agent是具有自主决策能力的智能体']，我来回答：RAG和Agent有什么区别？...
置信度: 0.62 工具: ['decompose_question', 'retrieve_knowledge']

=== 评估指标 ===
准确率: 80.0%
平均时间: 1.23s
多跳解决率: 66.7%
工具准确率: 100.0%
注：由于模拟模型生成置信度是随机的，实际运行结果会有所差异。

扩展建议
复杂决策：可引入更多工具（如计算器、API 调用、代码执行器），并增加工具选择策略（基于 LLM 的推理）。

记忆增强：利用 self.memory 实现对话历史感知，改进多轮问答质量。

并行检索：对于分解后的子问题，可使用线程池并行调用检索工具，降低延迟。

许可证
MIT License。可自由修改和分发，但请保留原始版权声明。

致谢
本项目受 LangChain 和 ReAct 范式启发，仅用于学习与演示动态 RAG Agent 的核心设计思想。
