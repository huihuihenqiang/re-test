import os
import torch
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorDBGenerator:
    def __init__(self, use_gpu: bool = True):
        # 硬件配置
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        print(f"使用设备: {self.device}")

        # OCR配置
        self.tesseract_path = r"D:\\ocr\\tesseract.exe"  # 修改为你的实际路径
        self.ocr_lang = "chi_sim"  # 中文OCR

        # 文本处理配置
        self.chunk_size = 700  # 文本分块大小
        self.chunk_overlap = 100  # 分块重叠量

        # 嵌入模型配置
        self.embedding_model = "shibing624/text2vec-base-chinese"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': self.device},  # 使用GPU加速
            encode_kwargs={
                'batch_size': 32,  # 批处理大小
                'normalize_embeddings': True
            }
        )

        # 初始化组件
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "。", "！", "？", "；"],
            length_function=len,
            add_start_index=True
        )

    def process_pdf(self, pdf_path: str, db_path: str = "pdf_vector_db"):
        """处理PDF文件生成向量数据库"""
        if os.path.exists(db_path):
            print(f"检测到现有数据库，直接加载: {db_path}")
            return Chroma(persist_directory=db_path, embedding_function=self.embeddings)

        print(f"开始处理PDF文件: {pdf_path}")
        documents = []

        with fitz.open(pdf_path) as pdf:
            for page_num in range(len(pdf)):
                page = pdf[page_num]

                # 尝试提取文本
                text = page.get_text()

                # 验证文本有效性
                if not self._is_valid_chinese(text):
                    # 转换为图像进行OCR
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    text = pytesseract.image_to_string(img, lang=self.ocr_lang)

                # 文本清洗和分块
                clean_text = self._clean_text(text)
                page_docs = self.text_splitter.create_documents([clean_text])

                # 添加元数据
                for doc in page_docs:
                    doc.metadata.update({
                        "source": pdf_path,
                        "page": page_num + 1,
                        "content_type": "text"
                    })
                documents.extend(page_docs)

        # 创建向量数据库
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=db_path
        )
        vector_db.persist()
        print(f"PDF向量数据库已保存至: {db_path}")
        return vector_db

    def process_txt(self, txt_path: str, db_path: str = "txt_vector_db"):
        """处理TXT文件生成向量数据库"""
        if os.path.exists(db_path):
            print(f"检测到现有数据库，直接加载: {db_path}")
            return Chroma(persist_directory=db_path, embedding_function=self.embeddings)

        print(f"开始处理TXT文件: {txt_path}")
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        # 文本清洗和分块
        clean_text = self._clean_text(text)
        documents = self.text_splitter.create_documents([clean_text])

        # 添加元数据
        for doc in documents:
            doc.metadata.update({
                "source": txt_path,
                "content_type": "text"
            })

        # 创建向量数据库
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=db_path
        )
        vector_db.persist()
        print(f"TXT向量数据库已保存至: {db_path}")
        return vector_db

    def _is_valid_chinese(self, text: str) -> bool:
        """验证中文内容有效性"""
        chinese_chars = sum('\u4e00' <= c <= '\u9fff' for c in text)
        return chinese_chars / max(len(text), 1) > 0.3  # 中文字符占比超过30%

    def _clean_text(self, text: str) -> str:
        """中文文本清洗"""
        # 移除特殊字符但保留中文标点
        text = re.sub(r'[^\u4e00-\u9fa5，。！？；：“”‘’（）【】％‰、\n\sA-Za-z0-9]', '', text)
        # 合并多余空白
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


if __name__ == "__main__":
    # 初始化处理器（自动检测GPU）
    processor = VectorDBGenerator(use_gpu=True)

    # 处理PDF文件
    pdf_db = processor.process_pdf("book.pdf")

    # 处理TXT文件
    txt_db = processor.process_txt("trans.txt")

    # 示例检索
    print("\nPDF数据库检索示例:")
    results = pdf_db.similarity_search("心理学原理", k=2)
    for doc in results:
        print(f"[Page {doc.metadata['page']}] {doc.page_content[:100]}...")

    print("\nTXT数据库检索示例:")
    results = txt_db.similarity_search("认知行为", k=2)
    for doc in results:
        print(f"[{doc.metadata['source']}] {doc.page_content[:100]}...")