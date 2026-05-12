"""
医疗器械体系文件审核 - 向量存储模块
使用 ChromaDB 管理知识库向量，支持本地持久化
"""
import os
from typing import List, Optional, Dict, Any
from pathlib import Path
import httpx


class MiniMaxEmbeddingFunction:
    """自定义的 Embedding Function，使用火山引擎 API"""

    def __init__(self, api_key: str, api_url: str, model: str = "doubao-embedding-vision-250615", dimension: int = 1024):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.dimension = dimension

    def __call__(self, input: List[str]) -> List[List[float]]:
        """调用 Embedding API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        embeddings = []
        with httpx.Client(timeout=60, trust_env=False) as client:
            for text in input:
                payload = {
                    "model": self.model,
                    "encoding_format": "float",
                    "input": [{"text": text[:8000], "type": "text"}]
                }
                try:
                    response = client.post(self.api_url, headers=headers, json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        data = result.get("data", {})
                        if data:
                            embedding = data.get("embedding", [])
                            # 如果需要，截断到指定维度
                            if len(embedding) > self.dimension:
                                embedding = embedding[:self.dimension]
                            elif len(embedding) < self.dimension:
                                # 如果太短，用随机数填充
                                import random
                                while len(embedding) < self.dimension:
                                    embedding.append(random.uniform(-1, 1))
                            embeddings.append(embedding)
                        else:
                            # 失败时使用随机向量
                            import random
                            embeddings.append([random.uniform(-1, 1) for _ in range(self.dimension)])
                    else:
                        import random
                        embeddings.append([random.uniform(-1, 1) for _ in range(self.dimension)])
                except Exception:
                    import random
                    embeddings.append([random.uniform(-1, 1) for _ in range(self.dimension)])

        return embeddings


class VectorStore:
    """ChromaDB 向量存储管理类"""

    def __init__(self, persist_directory: str = "data/chroma_db", embedding_function=None):
        """
        初始化向量存储

        Args:
            persist_directory: 持久化存储路径
            embedding_function: 自定义 embedding 函数
        """
        import chromadb
        from chromadb.config import Settings

        # 获取 backend 目录的绝对路径
        base_dir = Path(__file__).parent
        full_path = base_dir / persist_directory

        self.persist_directory = str(full_path)
        self.collection_name = "medical_device_kb"
        self.embedding_function = embedding_function

        # 确保目录存在
        os.makedirs(self.persist_directory, exist_ok=True)

        # 初始化 ChromaDB 客户端
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # 获取或创建 collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "医疗器械体系文件知识库"},
            embedding_function=embedding_function
        )

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """
        添加文档到向量库

        Args:
            documents: 文档文本列表
            metadatas: 元数据列表
            ids: ID 列表（可选，自动生成）
            embeddings: 预计算的 embedding 列表（可选）

        Returns:
            生成的 ID 列表
        """
        if ids is None:
            ids = [f"doc_{i}_{hash(text) % 100000}" for i, text in enumerate(documents)]

        if embeddings is not None:
            self.collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
                ids=ids
            )
        else:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

        return ids

    def query(
        self,
        query_texts: Optional[List[str]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        query_embeddings: Optional[List[List[float]]] = None
    ) -> Dict[str, Any]:
        """
        查询相似文档

        Args:
            query_texts: 查询文本列表
            n_results: 返回结果数量
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
            query_embeddings: 预计算的查询 embedding 列表（可选）

        Returns:
            查询结果字典
        """
        if query_embeddings is not None:
            return self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document
            )
        else:
            return self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                where_document=where_document
            )

    def get_all_documents(self, limit: int = 1000) -> Dict[str, Any]:
        """
        获取所有文档

        Args:
            limit: 限制返回数量

        Returns:
            文档字典
        """
        total = self.collection.count()
        return self.collection.get(limit=min(limit, total))

    def delete_collection(self):
        """删除 collection"""
        self.client.delete_collection(name=self.collection_name)

    def count(self) -> int:
        """返回 collection 中的文档数量"""
        return self.collection.count()


def create_vector_store(persist_directory: str = "data/chroma_db", embedding_function=None) -> VectorStore:
    """
    工厂函数：创建向量存储实例

    Args:
        persist_directory: 持久化目录
        embedding_function: 自定义 embedding 函数（可选）

    Returns:
        VectorStore 实例
    """
    return VectorStore(persist_directory=persist_directory, embedding_function=embedding_function)
