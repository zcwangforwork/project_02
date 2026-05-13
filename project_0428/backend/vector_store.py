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
            embedding_function: 自定义 embedding 函数（用于查询时生成 query embedding）
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

        # 获取主 collection（不传 embedding_function，避免初始化时加载 HNSW 索引）
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "医疗器械体系文件知识库"}
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

    def _get_query_embeddings(self, query_texts: List[str]) -> Optional[List[List[float]]]:
        """
        使用 embedding_function 将查询文本转为向量

        Args:
            query_texts: 查询文本列表

        Returns:
            embedding 列表，如果 embedding_function 不可用则返回 None
        """
        if self.embedding_function is None:
            return None
        try:
            return self.embedding_function(query_texts)
        except Exception as e:
            print(f"生成 query embedding 失败: {e}")
            return None

    def query(
        self,
        query_texts: Optional[List[str]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        query_embeddings: Optional[List[List[float]]] = None
    ) -> Dict[str, Any]:
        """
        查询相似文档（自动合并 v2 collection 和主 collection 的结果）

        优先查询 v2 collection（数据量大且可加载），主 collection 作为回退。
        使用 embedding_function 手动生成 query embedding，避免 ChromaDB 默认
        embedding function 的维度不匹配问题。

        Args:
            query_texts: 查询文本列表
            n_results: 返回结果数量
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
            query_embeddings: 预计算的查询 embedding 列表（可选）

        Returns:
            查询结果字典（合并两个 collection 的结果，按距离排序）
        """
        # 如果没有预计算的 embedding，使用 embedding_function 生成
        if query_embeddings is None and query_texts is not None:
            query_embeddings = self._get_query_embeddings(query_texts)

        # 构建 query_embeddings 优先的查询参数
        if query_embeddings is not None:
            query_kwargs = {
                "query_embeddings": query_embeddings,
                "n_results": n_results,
                "where": where,
                "where_document": where_document,
            }
        else:
            # 如果没有 embedding 也没有 embedding_function，回退到 query_texts
            query_kwargs = {
                "query_texts": query_texts,
                "n_results": n_results,
                "where": where,
                "where_document": where_document,
            }

        # 优先查询 v2 collection（数据量更大，HNSW 索引可加载）
        v2_results = None
        try:
            v2_collection = self.client.get_collection("medical_device_kb_v2")
            v2_count = v2_collection.count()
            if v2_count > 0:
                v2_results = v2_collection.query(**query_kwargs)
        except Exception as e:
            print(f"查询 v2 collection 失败: {e}")

        # 尝试查询主 collection（可能因 HNSW 内存不足而失败）
        main_results = None
        try:
            main_results = self.collection.query(**query_kwargs)
        except RuntimeError as e:
            if "Not enough memory" in str(e):
                print(f"主 collection 查询跳过（HNSW 内存不足）")
            else:
                raise
        except Exception as e:
            print(f"主 collection 查询失败: {e}")

        # 如果都没有结果，返回空
        if v2_results is None and main_results is None:
            return {
                'ids': [[]],
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }

        # 只有一个有结果时直接返回
        if v2_results is None or not v2_results.get('documents') or not v2_results['documents'][0]:
            return main_results
        if main_results is None or not main_results.get('documents') or not main_results['documents'][0]:
            return v2_results

        # 合并两个 collection 的结果，按距离排序
        return self._merge_query_results(main_results, v2_results, n_results)

    def _merge_query_results(self, main_results: Dict, v2_results: Dict, n_results: int) -> Dict:
        """
        合并两个 collection 的查询结果，按距离排序取 top-n

        Args:
            main_results: 主 collection 查询结果
            v2_results: v2 collection 查询结果
            n_results: 最终返回的结果数量

        Returns:
            合并后的查询结果字典
        """
        merged = {
            'ids': [[]],
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }

        # 收集所有结果
        all_items = []

        # 主 collection 结果
        for i in range(len(main_results.get('ids', [[]])[0])):
            item = {'id': main_results['ids'][0][i]}
            if 'documents' in main_results and main_results['documents']:
                item['document'] = main_results['documents'][0][i]
            if 'metadatas' in main_results and main_results['metadatas']:
                item['metadata'] = main_results['metadatas'][0][i]
            else:
                item['metadata'] = {}
            if 'distances' in main_results and main_results['distances']:
                item['distance'] = main_results['distances'][0][i]
            else:
                item['distance'] = float('inf')
            all_items.append(item)

        # v2 collection 结果
        for i in range(len(v2_results.get('ids', [[]])[0])):
            item = {'id': v2_results['ids'][0][i]}
            if 'documents' in v2_results and v2_results['documents']:
                item['document'] = v2_results['documents'][0][i]
            if 'metadatas' in v2_results and v2_results['metadatas']:
                item['metadata'] = v2_results['metadatas'][0][i]
            else:
                item['metadata'] = {}
            if 'distances' in v2_results and v2_results['distances']:
                item['distance'] = v2_results['distances'][0][i]
            else:
                item['distance'] = float('inf')
            all_items.append(item)

        # 按距离排序
        all_items.sort(key=lambda x: x['distance'])

        # 取 top-n
        for item in all_items[:n_results]:
            merged['ids'][0].append(item['id'])
            merged['documents'][0].append(item.get('document', ''))
            merged['metadatas'][0].append(item.get('metadata', {}))
            merged['distances'][0].append(item.get('distance', float('inf')))

        return merged

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
