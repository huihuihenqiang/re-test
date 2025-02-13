import os
import re
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import snapshot_download
import sys
from datetime import datetime


#todo:除了检索知识库的内容，还要检索到知识库对应书本的内容。

class Tee:
    """双输出流类（安全关闭版本）"""

    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8')
        self.original_stdout = sys.stdout
        sys.stdout = self

    def write(self, message):
        # 确保原始输出正常
        self.original_stdout.write(message)
        # 文件未关闭时写入
        if self.file and not self.file.closed:
            self.file.write(message)

    def flush(self):
        self.original_stdout.flush()
        if self.file and not self.file.closed:
            self.file.flush()

    def close(self):
        # 恢复标准输出
        sys.stdout = self.original_stdout
        # 安全关闭文件
        if self.file and not self.file.closed:
            self.flush()  # 确保缓冲区写入
            self.file.close()


class ExperimentConfig:
    """实验参数配置"""

    def __init__(self):
        # 修改后的咨询师参数
        self.counselor_params = {
            "model": "deepseek-chat",
            "openai_api_base": "https://api.deepseek.com",  # 改为openai_api_base
            "openai_api_key": "sk-b903cbc42ff549e8a4849c2ce8d9d5b9",  # 改为openai_api_key
        }

        # 修改后的来访者参数
        self.client_params = {
            "model": "moonshot-v1-128k",
            "openai_api_base": "https://api.moonshot.cn/v1",
            "openai_api_key": "sk-cqxUL0HSqcH8MHFddrdHLbSiHylLF3mEcdCJaegQVIW1nVY5"
        }

        # 本地嵌入模型配置（与VectorDBGenerator一致）
        self.embedding_params = {
            "model_name": "shibing624/text2vec-base-chinese",  # 这里使用您本地已经下载好的模型
            "model_kwargs": {'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            "encode_kwargs": {
                'batch_size': 32,
                'normalize_embeddings': True
            }
        }

        # 知识库路径配置
        self.kb_paths = {
            "pdf": "pdf_vector_db",
            "processed": "txt_vector_db"
        }

        # 设置Hugging Face缓存路径
        self.hf_cache_dir = r"D:\\huggingface_cache"
        if not os.path.exists(self.hf_cache_dir):
            os.makedirs(self.hf_cache_dir)
        os.environ["HF_HOME"] = self.hf_cache_dir

        # 检查是否存在本地缓存的模型，没有则下载
        self.download_model_if_needed()

    def download_model_if_needed(self):
        """检查本地缓存是否已有该模型，如果没有则下载"""
        model_path = os.path.join(self.hf_cache_dir, "models--shibing624--text2vec-base-chinese")
        if not os.path.exists(model_path):
            print("模型未找到，开始下载...")
            snapshot_download(repo_id="shibing624/text2vec-base-chinese", local_dir=self.hf_cache_dir)
            print(f"模型已下载到: {model_path}")
        else:
            print("模型已存在，直接使用本地缓存。")


# 修改CounselingSystem中的知识库初始化
class CounselingSystem:
    def __init__(self, config: ExperimentConfig, kb_type: str = "none"):
        self.config = config
        self.kb_type = kb_type
        self.dialogue_history = []
        self.retrieval_records = []
        self.client_persona = None
        self.vectorstore = None  # 添加vectorstore属性初始化

        # 初始化模型
        self.counselor_llm = ChatOpenAI(**config.counselor_params)
        self.client_llm = ChatOpenAI(**config.client_params)

        # 初始化知识库
        self.vectorstore = self._init_knowledge(kb_type)

        # 初始化提示模板
        self._init_prompts()

    def _init_knowledge(self, kb_type: str) -> Optional[Chroma]:
        """加载预处理的知识库"""
        if kb_type == "none":
            return None

        if kb_type not in self.config.kb_paths:
            raise ValueError(f"不支持的知识库类型: {kb_type}")

        db_path = self.config.kb_paths[kb_type]
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"知识库 {kb_type} 未找到，请先运行preprocess.py预处理\n"
                f"预期路径: {os.path.abspath(db_path)}"
            )

        # 使用HuggingFace嵌入模型
        embeddings = HuggingFaceEmbeddings(**self.config.embedding_params)

        return Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )

    def _init_prompts(self):
        """初始化双角色提示模板"""
        # 咨询师模板
        counselor_base = """You are a helpful, respectful, honest, and knowledgeable licensed therapist.You are in psychotherapy with a client.请用中文回答。 
        Please use CBT therapy to heal the client through dialogue. Based on the information gathered through the conversation, evaluate whether the client may be dealing with emotional, behavioral, or developmental issues, 
        and based on the client's response, choose appropriate assessment tools to further explore their situation, in combination with CBT therapy to try to cure the client or make the client's condition better."""
        if self.kb_type != "none":
            counselor_base += """

            根据以下咨询原则进行回应：
            {context}  """

        self.counselor_template = PromptTemplate.from_template(
            counselor_base + "\n当前问题：{question}\n咨询师回应："
        )

        # 来访者模板
        self.client_template = PromptTemplate.from_template(
            """根据你的人设背景和对话历史，扮演其中的病人，用自然的口语继续对话。我是一名心理咨询师，我会和你展开对话。请用中文回答：

            人设背景：
            {persona}

            对话历史（最近3轮）：
            {history}

            请用1-3句话表达你的想法或感受："""
        )

    def set_client_persona(self, persona: str):
        """设置来访者人设"""
        self.client_persona = persona
        # 初始化人设后清空历史
        self.dialogue_history = []

    def run_dialogue(self, max_rounds: int = 5):
        """执行完整对话流程"""
        if not self.client_persona:
            raise ValueError("必须使用set_client_persona()设置来访者人设")

        # 生成初始陈述
        client_msg = self._generate_client_response(init=True)
        self._update_history("client", client_msg)

        round_count = 0
        while round_count < max_rounds:
            # 咨询师回应
            start_time = time.time()
            counselor_response, contexts = self._generate_counselor_response(client_msg)
            latency = time.time() - start_time

            self._update_history("counselor", counselor_response)
            self.retrieval_records.append({
                "round": round_count,
                "latency": latency,
                "contexts": contexts
            })

            # 打印每轮对话并输出检索到的信息
            print(f"\n【轮次 {round_count + 1}】")
            print(f"CLIENT: {client_msg}")
            print(f"COUNSELOR: {counselor_response}")

            # 输出检索到的文档内容
            if contexts:
                print("【检索到的相关文档】")
                for ctx in contexts:
                    print(f"来源: {ctx['source']}, 页码: {ctx['page'] if ctx['page'] != 'N/A' else 'N/A'}")
                    print(f"内容: {ctx['content'][:150]}...")  # 显示文档内容的前100个字符

            # 来访者回应
            client_msg = self._generate_client_response()
            self._update_history("client", client_msg)

            round_count += 1

            # 提前终止条件
            if "结束对话" in client_msg:
                break

    def _generate_counselor_response(self, query: str) -> Tuple[str, List[Dict]]:
        """生成咨询师回应"""
        contexts = []
        retrieval_info = ""

        if self.vectorstore:
            try:
                # 进行检索
                docs = self.vectorstore.similarity_search_with_relevance_scores(
                    query,
                    k=2,
                    score_threshold=0.6  # 调整阈值以确保检索到相关文档
                )

                # 获取检索到的文档，并构建返回信息
                contexts = [{
                    "content": doc.page_content,
                    "score": float(score),
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "unknown"),
                    "content_type": doc.metadata.get("content_type", "text")  # 添加content_type
                } for doc, score in docs]

                # 拼接检索到的文档信息，用于构建回复
                retrieval_info = "\n".join(
                    f"【检索信息】\n"
                    f"来源: {ctx['source']}, "
                    f"页码: {ctx['page'] if ctx['content_type'] == 'text' else 'N/A'}\n"
                    f"内容: {ctx['content'][:100]}..."  # 截取部分内容展示
                    for ctx in contexts
                )

            except Exception as e:
                print(f"知识检索失败: {str(e)}")
                contexts = []

        # 构建咨询师的回复
        counselor_response = self._generate_counselor_message(query, retrieval_info)

        return counselor_response, contexts

    def _generate_counselor_message(self, query: str, retrieval_info: str) -> str:
        """构建咨询师回复"""
        counselor_base = "You are a helpful, respectful, honest, and knowledgeable licensed therapist.You are in psychotherapy with a client.请用中文回答。 Please use CBT therapy to heal the client through dialogue. Based on the information gathered through the conversation, evaluate whether the client may be dealing with emotional, behavioral, or developmental issues, and based on the client's response, choose appropriate assessment tools to further explore their situation, in combination with CBT therapy to try to cure the client or make the client's condition better."

        # 使用检索结果构建回应
        counselor_message = f"{counselor_base}\n\n根据以下内容提供帮助:\n{retrieval_info}\n\n当前问题：{query}\n咨询师回应："

        # 生成最终的回复内容
        try:
            response = self.counselor_llm.invoke(counselor_message).content
            return response
        except Exception as e:
            print(f"生成咨询师回应失败: {str(e)}")
            return "我理解您的感受，能具体说说发生了什么吗？"

    def _generate_client_response(self, init: bool = False) -> str:
        """生成来访者陈述"""
        history_str = "\n".join(
            f"{role}: {content}"
            for role, content in self.dialogue_history[-3:]  # 保持最近3轮
        ) if not init else "无对话历史"

        prompt = self.client_template.format(
            persona=self.client_persona,
            history=history_str
        )

        try:
            response = self.client_llm.invoke(prompt).content
            # 清理响应中的多余引号
            return response.strip('"').strip()
        except Exception as e:
            print(f"生成来访者回应失败: {str(e)}")
            return "我觉得这个问题可能不太重要，我们换个话题吧。"

    def _update_history(self, role: str, content: str):
        """更新对话历史"""
        self.dialogue_history.append((role, content))


#
# # 修改EnhancedEvaluator中的嵌入模型使用
# class EnhancedEvaluator:
#     def __init__(self, config: ExperimentConfig):
#         self.config = config
#
#     def evaluate_session(self, system: CounselingSystem) -> Dict:
#         """评估生成质量"""
#         history = system.dialogue_history
#         retrieval_data = self._evaluate_retrieval(system.retrieval_records)  # 添加检索数据
#         generation_data = self._evaluate_generation(history)
#         return {
#             "retrieval": retrieval_data,
#             "generation": generation_data
#         }
#
#     def _evaluate_retrieval(self, retrieval_records: List[Dict]) -> Dict:
#         """评估检索质量"""
#         if not retrieval_records:
#             return {"average_latency": 0}
#         avg_latency = np.mean([record["latency"] for record in retrieval_records])
#         return {"average_latency": avg_latency}
#
#     def _evaluate_generation(self, history: List[Tuple]) -> Dict:
#         """评估生成质量"""
#         counselor_responses = [content for role, content in history if role == "counselor"]
#         client_responses = [content for role, content in history if role == "client"]
#
#         # 长度分析
#         counselor_lens = [len(r) for r in counselor_responses]
#         client_lens = [len(r) for r in client_responses]
#
#         # 语义连贯性分析（使用本地模型）
#         embeddings = HuggingFaceEmbeddings(**self.config.embedding_params)
#
#         # 分批处理避免内存溢出
#         batch_size = 32
#         counselor_vecs = []
#         for i in range(0, len(counselor_responses), batch_size):
#             batch = counselor_responses[i:i + batch_size]
#             counselor_vecs.extend(embeddings.embed_documents(batch))
#
#         client_vecs = []
#         for i in range(0, len(client_responses), batch_size):
#             batch = client_responses[i:i + batch_size]
#             client_vecs.extend(embeddings.embed_documents(batch))
#
#         # 计算相邻对话的相似度
#         counselor_coherence = np.mean([
#             cosine_similarity([counselor_vecs[i]], [counselor_vecs[i + 1]])[0][0]
#             for i in range(len(counselor_vecs) - 1)
#         ]) if len(counselor_vecs) > 1 else 0
#
#         return {
#             "counselor_avg_len": np.mean(counselor_lens),
#             "client_avg_len": np.mean(client_lens),
#             "counselor_coherence": counselor_coherence,
#             "turn_ratio": len(counselor_responses) / len(client_responses) if client_responses else 0
#         }
class EnhancedEvaluator:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(**config.embedding_params)

        # 初始化LLM评估器
        self.faithfulness_evaluator = ChatOpenAI(
            temperature=0,
            model="moonshot-v1-128k",
            openai_api_key=config.client_params["openai_api_key"],
            openai_api_base=config.client_params["openai_api_base"]
        )

        # 评估模板
        self.faithfulness_prompt = PromptTemplate.from_template("""
            作为心理咨询评估专家，请严格评估回复的忠实度：
            - 必须基于提供的咨询原则（0-10分）
            - 不得包含矛盾或虚构内容

            咨询原则：
            {context}

            咨询师回复：
            {answer}

            直接给出0-10的整数分数，不要解释。
            """)

        # 领域关键词库
        self.domain_keywords = [
            '认知重构', '行为', '暴露疗法', '自动思维',
            '共情', 'CBT', '倾听技巧', '情感反映', '记录'
        ]

    def evaluate_session(self, system: CounselingSystem) -> Dict:
        """执行完整评估"""
        history = system.dialogue_history
        return {
            "retrieval": self._evaluate_retrieval(system.retrieval_records),
            "generation": self._evaluate_generation(history, system.retrieval_records)
        }

    def _evaluate_retrieval(self, records: List[Dict]) -> Dict:
        """评估检索质量"""
        if not records:
            return {"recall@1": 0.0, "recall@3": 0.0,
                    "precision@1": 0.0, "precision@3": 0.0}

        total = len(records)
        recall_1, recall_3 = 0.0, 0.0
        precision_1, precision_3 = 0.0, 0.0

        for record in records:
            # 假设每个查询有1个相关文档（需真实数据改进）
            retrieved = len(record.get('contexts', []))
            relevant = 1  # 简化假设

            # Recall计算
            recall_1 += min(retrieved, 1) / relevant
            recall_3 += min(retrieved, 3) / relevant

            # Precision计算
            precision_1 += min(1, retrieved) / 1
            precision_3 += min(3, retrieved) / 3

        return {
            "recall@1": recall_1 / total,
            "recall@3": recall_3 / total,
            "precision@1": precision_1 / total,
            "precision@3": precision_3 / total
        }

    def _evaluate_generation(self, history: List[Tuple], records: List[Dict]) -> Dict:
        """评估生成质量"""
        counselor_responses = [c for _, c in history if _ == "counselor"]

        # 基础指标
        avg_len = np.mean([len(r) for r in counselor_responses]) if counselor_responses else 0
        vecs = self._batch_embed(counselor_responses)
        coherence = self._calculate_coherence(vecs)

        # 高级指标
        faithfulness = []
        integration = []
        for idx, resp in enumerate(counselor_responses):
            # 忠实度评估
            if idx < len(records) and records[idx]['contexts']:
                contexts = [c['content'] for c in records[idx]['contexts']]
                faithfulness.append(self._eval_faithfulness(resp, contexts))

            # 信息整合评估
            integration.append(
                sum(1 for kw in self.domain_keywords if kw in resp) / len(self.domain_keywords)
            )

        return {
            "counselor_avg_len": avg_len,
            "counselor_coherence": coherence,
            "faithfulness": np.mean(faithfulness) if faithfulness else 0.0,
            "information_integration": np.mean(integration) if integration else 0.0
        }

    def _eval_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """使用LLM评估忠实度"""
        try:
            context_str = "\n".join([c[:500] for c in contexts])[:2000]
            prompt = self.faithfulness_prompt.format(
                context=context_str,
                answer=answer[:1000]
            )
            response = self.faithfulness_evaluator.invoke(prompt).content
            return float(re.search(r"\d+", response).group()) / 10  # 转换为0-1范围
        except Exception as e:
            print(f"忠实度评估失败: {str(e)}")
            return 0.0

    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        """批量编码文本"""
        return [self.embeddings.embed_documents([t])[0] for t in texts] if texts else []

    def _calculate_coherence(self, vecs: List[List[float]]) -> float:
        """计算连贯性得分"""
        if len(vecs) < 2:
            return 0.0
        return np.mean([cosine_similarity([vecs[i]], [vecs[i + 1]])[0][0]
                        for i in range(len(vecs) - 1)])


if __name__ == "__main__":
    config = ExperimentConfig()

    # 日志配置
    log_dir = "experiment_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    tee = Tee(log_file)

    try:
        print(f"=== 实验开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

        # 测试配置
        PERSONA = """22男性，同性恋。重度抑郁症：
            - 我四个月前被父母赶出家门，上个月又失业了。之前挺抑郁的，现在因为交了男朋友，
            没那么绝望和抑郁了。但还是会因性取向和父母的拒绝，感到愧疚和羞耻。"""

        systems = {
            "no_kb": CounselingSystem(config, "none"),
            "pdf_kb": CounselingSystem(config, "pdf"),
            "txt_kb": CounselingSystem(config, "processed")
        }

        evaluator = EnhancedEvaluator(config)
        results = {}

        for name, sys in systems.items():
            print(f"\n{'=' * 20} {name.upper()} 版本 {'=' * 20}")
            sys.set_client_persona(PERSONA)
            sys.run_dialogue(max_rounds=3)
            results[name] = evaluator.evaluate_session(sys)

        # 打印评估报告
        print("\n=== 综合评估结果 ===")
        for name, metrics in results.items():
            print(f"\n【{name.upper()}】")

            print("检索质量:")
            print(f"  Recall@1: {metrics['retrieval']['recall@1']:.2f}")
            print(f"  Recall@3: {metrics['retrieval']['recall@3']:.2f}")
            print(f"  Precision@1: {metrics['retrieval']['precision@1']:.2f}")
            print(f"  Precision@3: {metrics['retrieval']['precision@3']:.2f}")

            print("生成质量:")
            print(f"  平均回复长度: {metrics['generation']['counselor_avg_len']:.1f}字")
            print(f"  连贯性: {metrics['generation']['counselor_coherence']:.2f}")
            print(f"  忠实度: {metrics['generation']['faithfulness']:.2f}")
            print(f"  信息整合: {metrics['generation']['information_integration']:.2f}")

    finally:
        tee.close()
        print(f"\n日志文件已保存至: {os.path.abspath(log_file)}")
