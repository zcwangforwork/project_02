#!/usr/bin/env python
"""专门检查医械标准库内容"""
import os
import sys
from pathlib import Path
from collections import defaultdict

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from vector_store import create_vector_store

def main():
    # 使用绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("医械标准库内容检查")
    print("=" * 70)

    # 创建向量存储
    vs = create_vector_store(persist_directory="data/chroma_db")

    print(f"\n向量库总文档数: {vs.count()}")

    # 获取所有文档
    print("\n正在获取文档...")
    all_docs = vs.get_all_documents(limit=20000)

    print(f"实际获取文档数: {len(all_docs.get('ids', []))}")

    # 统计分类
    categories = defaultdict(int)
    medical_docs = defaultdict(int)

    ids = all_docs.get('ids', [])
    metadatas = all_docs.get('metadatas', [])
    documents = all_docs.get('documents', [])

    for i, meta in enumerate(metadatas):
        if meta:
            category = meta.get('category', '未知')
            categories[category] += 1

            if '医械' in category or '标准库' in category:
                source = meta.get('source', '未知')
                medical_docs[source] += 1

    print("\n" + "=" * 70)
    print("所有分类统计:")
    print("=" * 70)
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} 个文本块")

    print("\n" + "=" * 70)
    print("医械标准库文档统计:")
    print("=" * 70)

    if medical_docs:
        total = sum(medical_docs.values())
        print(f"\n共找到 {len(medical_docs)} 个文件，{total} 个文本块\n")

        for source, count in sorted(medical_docs.items()):
            print(f"  {source}: {count} 个文本块")

        # 显示一些预览
        print("\n" + "=" * 70)
        print("文档预览:")
        print("=" * 70)

        preview_count = 0
        for i, meta in enumerate(metadatas):
            if preview_count >= 10:
                break

            if meta:
                category = meta.get('category', '')
                if '医械' in category or '标准库' in category:
                    source = meta.get('source', '未知')
                    print(f"\n{preview_count + 1}. {source}")
                    print(f"   分类: {category}")
                    if i < len(documents) and documents[i]:
                        preview = documents[i][:150].replace('\n', ' ')
                        print(f"   预览: {preview}...")
                    preview_count += 1

    else:
        print("  未找到医械标准库文档")

    print("\n" + "=" * 70)
    print("检查完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()
