## 知识图谱
搭建知识图谱使用的是neo4j,查询使用的是cypher。
## agent
## 📝 文件概述

### 1. `embedding.py` 📄
- **功能**：从 **PDF** 和 **TXT** 文件中提取文本，处理中文文本（包括清洗和验证），并生成 **向量数据库**。
- **流程**：
  1. 提取PDF/TXT文件中的文本（图片中的文字通过OCR提取）。
  2. 清洗并验证中文文本的有效性。
  3. 使用 **shibing624/text2vec-base-chinese** 模型将文本转化为向量。
  4. 将向量存储在向量数据库中。

### 2. `retrieval.py` 🔍
- **功能**：实现 **RAG** 模型，检索向量数据库中的内容，并评估检索和生成的质量。
- **流程**：
  1. 根据查询检索相关内容。
  2. 评估检索质量（相关性）。
  3. 使用检索到的内容生成响应，并评估生成质量。
  4. 可以显示索引信息，帮助理解检索过程。

结果展示：

![图片](https://github.com/user-attachments/assets/773429cd-5a79-4a79-96ef-132b7ee13578)

评估展示：

![图片](https://github.com/user-attachments/assets/37c984f3-0f0c-4449-846f-26f553d8f03c)

