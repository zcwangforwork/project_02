#!/usr/bin/env python
"""检查向量库中的文档内容"""
import os
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from vector_store import create_vector_store, MiniMaxEmbeddingFunction

def main():
    # 使用绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "data", "chroma_db")

    # API配置
    API_KEY = "ark-6c047509-3ac1-4689-b796-47363425012c-3112a"
    EMBEDDING_URL = "https://ark.cn-beijing.volces.com/api/coding/v3/embeddings/multimodal"
    EMBEDDING_MODEL = "doubao-embedding-vision-250615"

    print("=" * 70)
    print("检查向量库内容")
    print("=" * 70)

    # 创建 embedding function
    embedding_function = MiniMaxEmbeddingFunction(
        api_key=API_KEY,
        api_url=EMBEDDING_URL,
        model=EMBEDDING_MODEL,
        dimension=1024
    )

    # 创建向量存储
    vs = create_vector_store(persist_directory=db_path, embedding_function=embedding_function)

    print(f"\n向量库中文档总数: {vs.count()}")

    # 获取所有文档
    print("\n正在获取所有文档（可能需要一些时间）...")
    all_docs = vs.get_all_documents(limit=10000)

    print(f"\n实际获取到的文档数: {len(all_docs.get('ids', []))}")

    # 统计来源文件
    sources = {}
    categories = {}

    metadatas = all_docs.get('metadatas', [])
    for meta in metadatas:
        if meta:
            source = meta.get('source', '未知')
            category = meta.get('category', '未知')

            sources[source] = sources.get(source, 0) + 1
            categories[category] = categories.get(category, 0) + 1

    print("\n" + "=" * 70)
    print("文档来源统计:")
    print("=" * 70)
    for source, count in sorted(sources.items()):
        print(f"  {source}: {count} 个文本块")

    print("\n" + "=" * 70)
    print("分类统计:")
    print("=" * 70)
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count} 个文本块")

    # 检查医械标准库相关的文档
    print("\n" + "=" * 70)
    print("医械标准库相关文档:")
    print("=" * 70)

    medical_standard_docs = []
    for source in sources.keys():
        if '医械' in source or '标准' in source or 'YY' in source or 'GB' in source or 'ISO' in source:
            medical_standard_docs.append((source, sources[source]))

    if medical_standard_docs:
        for source, count in sorted(medical_standard_docs):
            print(f"  {source}: {count} 个文本块")
        print(f"\n医械标准库相关文档总数: {sum(count for _, count in medical_standard_docs)} 个文本块")
    else:
        print("  未找到明确标注为医械标准库的文档")

    # 显示部分文档预览
    print("\n" + "=" * 70)
    print("文档预览（前10个）:")
    print("=" * 70)

    ids = all_docs.get('ids', [])
    documents = all_docs.get('documents', [])
    metadatas = all_docs.get('metadatas', [])

    for i in range(min(10, len(ids))):
        print(f"\n{i+1}. ID: {ids[i]}")
        if i < len(metadatas) and metadatas[i]:
            print(f"   来源: {metadatas[i].get('source', '未知')}")
            print(f"   分类: {metadatas[i].get('category', '未知')}")
        if i < len(documents) and documents[i]:
            preview = documents[i][:200].replace('\n', ' ')
            print(f"   内容预览: {preview}...")

    print("\n" + "=" * 70)
    print("检查完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()
