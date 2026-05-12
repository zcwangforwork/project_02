#!/usr/bin/env python
"""摄入文档目录中所有未摄入的文件，使用真实 Embedding API（低内存版本）"""
import os
import sys
import time
import gc
import httpx
from pathlib import Path
from collections import defaultdict

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from doc_processor import extract_text, chunk_text, get_file_metadata

# Embedding API 配置
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("MINIMAX_API_KEY") or "ark-6c047509-3ac1-4689-b796-47363425012c-3112a"
EMBEDDING_URL = "https://ark.cn-beijing.volces.com/api/coding/v3/embeddings/multimodal"
EMBEDDING_MODEL = "doubao-embedding-vision-250615"
EMBEDDING_DIM = 1024

# 低内存优化：复用 httpx Client，避免每次创建/销毁
_http_client = None

def _get_http_client():
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(timeout=60, trust_env=False)
    return _http_client


def get_embedding(text: str) -> list:
    """调用 Embedding API 获取文本向量"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "encoding_format": "float",
        "input": [{"text": text[:8000], "type": "text"}]
    }
    try:
        client = _get_http_client()
        response = client.post(EMBEDDING_URL, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            data = result.get("data", {})
            if data:
                embedding = data.get("embedding", [])
                if len(embedding) > EMBEDDING_DIM:
                    return embedding[:EMBEDDING_DIM]
                return embedding
    except Exception as e:
        print(f"  [Embedding API 错误] {e}")
    return None


def get_embeddings_batch(texts: list, max_retries: int = 3) -> list:
    """逐个获取 embeddings，失败自动重试（低内存：不缓存全部结果，边获取边存入）"""
    results = [None] * len(texts)
    for attempt in range(max_retries):
        remaining_indices = [i for i, r in enumerate(results) if r is None]
        if not remaining_indices:
            break
        for i in remaining_indices:
            emb = get_embedding(texts[i])
            if emb:
                results[i] = emb
            time.sleep(0.05)
        if attempt < max_retries - 1:
            failed = sum(1 for r in results if r is None)
            if failed > 0:
                print(f"  重试 {attempt+2}/{max_retries}，{failed} 个待重试")
                time.sleep(1)
    return results


def scan_directory(base_dir):
    """扫描目录下所有支持的文件"""
    supported_exts = {'.docx', '.pdf', '.txt', '.doc'}
    all_files = []

    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            ext = Path(filename).suffix.lower()
            if ext not in supported_exts:
                continue

            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, base_dir)
            all_files.append({
                'filename': filename,
                'full_path': full_path,
                'rel_path': rel_path,
                'directory': os.path.relpath(root, base_dir)
            })

    return all_files


def get_existing_sources():
    """从缓存文件加载已有的来源，如果不存在则从 ChromaDB 分页提取并缓存"""
    import json
    cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "existing_sources.json")

    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            sources = set(json.load(f))
        print(f"从缓存加载了 {len(sources)} 个已有来源")
        return sources

    # 缓存不存在，从 ChromaDB 分页提取（低内存：每次只取一批 metadata）
    print("缓存不存在，从 ChromaDB 分页提取已有来源...")
    import chromadb
    from chromadb.config import Settings

    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "data", "chroma_db")

    client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection("medical_device_kb")
    total = collection.count()

    sources = set()
    batch_size = 5000
    offset = 0
    while offset < total:
        result = collection.get(limit=batch_size, offset=offset, include=['metadatas'])
        for meta in result.get('metadatas', []):
            if meta:
                source = meta.get('source')
                if source:
                    sources.add(source)
        offset += batch_size
        print(f"  已提取 {min(offset, total)}/{total} 条...")

    # 释放 ChromaDB 客户端
    del client, collection
    gc.collect()

    # 保存缓存
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(sorted(sources), f, ensure_ascii=False)
    print(f"提取了 {len(sources)} 个已有来源，已保存到缓存")

    return sources


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    docs_dir = os.path.join(project_root, "develop_documents")

    print("=" * 80)
    print("文档摄入工具（使用真实 Embedding API）")
    print("=" * 80)
    print(f"文档目录: {docs_dir}")

    if not os.path.exists(docs_dir):
        print(f"\n错误: 目录不存在: {docs_dir}")
        return

    # 扫描目录
    print("\n正在扫描文档目录...")
    all_files = scan_directory(docs_dir)
    print(f"找到 {len(all_files)} 个文档文件")

    # 加载向量库（不传 embedding_function，加速加载）
    print("\n正在加载向量库...")
    import chromadb
    from chromadb.config import Settings

    db_path = os.path.join(base_dir, "data", "chroma_db")
    client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(
        name="medical_device_kb",
        metadata={"description": "医疗器械体系文件知识库"}
    )
    print(f"向量库当前文档数: {collection.count()} 个文本块")

    # 获取已摄入的文件
    print("\n正在检查已摄入的文件...")
    existing_sources = get_existing_sources()
    print(f"向量库中已有 {len(existing_sources)} 个来源文件")

    # 确定需要摄入的文件
    files_to_ingest = []
    for f in all_files:
        if f['filename'] not in existing_sources:
            files_to_ingest.append(f)

    print(f"\n需要摄入的新文件数: {len(files_to_ingest)}")

    if not files_to_ingest:
        print("\n所有文件都已摄入！")
        return

    # 按目录分组统计
    dir_stats = defaultdict(int)
    for f in files_to_ingest:
        dir_stats[f['directory']] += 1
    print(f"\n待摄入文件目录分布（前20）:")
    for d, count in sorted(dir_stats.items(), key=lambda x: -x[1])[:20]:
        print(f"  {d}: {count} 个文件")

    # 开始摄入
    print("\n开始摄入文档...")
    print("=" * 80)

    stats = {
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'total_chunks': 0,
        'api_errors': 0,
        'errors': []
    }

    start_time = time.time()
    # 低内存：分批提交大小，每处理一批文件后就提交并释放
    BATCH_COMMIT_SIZE = 20  # 每20个文件提交一次并触发GC

    # 低内存：暂存区，批量 add 到 ChromaDB 后立即清空
    pending_docs = []
    pending_metas = []
    pending_ids = []
    pending_embs = []

    def flush_pending():
        """将暂存区数据写入 ChromaDB 并释放内存"""
        nonlocal pending_docs, pending_metas, pending_ids, pending_embs
        if not pending_docs:
            return
        collection.add(
            documents=pending_docs,
            metadatas=pending_metas,
            ids=pending_ids,
            embeddings=pending_embs
        )
        chunk_count = len(pending_docs)
        pending_docs = []
        pending_metas = []
        pending_ids = []
        pending_embs = []
        gc.collect()
        return chunk_count

    for idx, file_info in enumerate(files_to_ingest):
        filename = file_info['filename']
        full_path = file_info['full_path']
        directory = file_info['directory']

        elapsed = time.time() - start_time
        if idx > 0:
            eta = elapsed / idx * (len(files_to_ingest) - idx)
            eta_str = f", 预计剩余 {eta/60:.0f} 分钟"
        else:
            eta_str = ""

        print(f"\n[{idx + 1}/{len(files_to_ingest)}] {filename}{eta_str}")
        print(f"  目录: {directory}")

        try:
            # 提取文本
            text = extract_text(full_path)
            if not text or len(text.strip()) < 20:
                print(f"  跳过: 文本过短或为空")
                stats['skipped'] += 1
                # 释放文本内存
                del text
                continue

            # 分块
            chunks = chunk_text(text, chunk_size=500, overlap=50)
            print(f"  分割成 {len(chunks)} 个文本块")

            # 低内存：立即释放原始文本
            del text
            gc.collect()

            # 获取元数据
            metadata = get_file_metadata(full_path)
            metadata['source'] = filename
            if directory == '.':
                metadata['category'] = "根目录"
            else:
                metadata['category'] = directory.replace(os.path.sep, '/')

            # 生成真实 embeddings
            print(f"  正在生成 {len(chunks)} 个 embedding 向量...")
            embeddings = get_embeddings_batch(chunks)

            # 检查哪些 embedding 失败了
            failed_emb = sum(1 for e in embeddings if e is None)
            if failed_emb > 0:
                stats['api_errors'] += failed_emb
                print(f"  警告: {failed_emb} 个 embedding 生成失败，将使用随机向量替代")
                import random
                embeddings = [e if e is not None else [random.uniform(-1, 1) for _ in range(EMBEDDING_DIM)] for e in embeddings]

            # 低内存：暂存而非立即 add，等批量提交
            for i, chunk in enumerate(chunks):
                chunk_id = f"ingest_{filename}_{idx}_{i}_{abs(hash(chunk)) % 1000000}"
                pending_docs.append(chunk)
                pending_metas.append(metadata.copy())
                pending_ids.append(chunk_id)
                pending_embs.append(embeddings[i])

            stats['success'] += 1
            stats['total_chunks'] += len(chunks)
            print(f"  成功！已暂存 {len(chunks)} 个文本块（待批量提交）")

            # 低内存：立即释放当前文件的 chunks 和 embeddings
            del chunks, embeddings
            gc.collect()

        except Exception as e:
            stats['failed'] += 1
            stats['errors'].append(f"{filename}: {str(e)}")
            print(f"  失败: {str(e)}")
            import traceback
            traceback.print_exc()

        # 每 BATCH_COMMIT_SIZE 个文件批量提交并释放内存
        if stats['success'] > 0 and stats['success'] % BATCH_COMMIT_SIZE == 0:
            flushed = flush_pending()
            print(f"\n  [批量提交] 已写入 {flushed} 个文本块到向量库")
            # 输出内存使用
            try:
                import psutil
                proc = psutil.Process(os.getpid())
                mem_mb = proc.memory_info().rss / 1024 / 1024
                print(f"  [内存] 当前进程 RSS: {mem_mb:.0f} MB")
            except ImportError:
                pass

        # 每50个文件显示一次进度汇总
        if (idx + 1) % 50 == 0:
            print(f"\n--- 进度: {idx + 1}/{len(files_to_ingest)}, "
                  f"成功: {stats['success']}, 失败: {stats['failed']}, "
                  f"跳过: {stats['skipped']}, 文本块: {stats['total_chunks']} ---")

        # 短暂暂停避免过快
        time.sleep(0.02)

    # 处理结束，提交剩余暂存数据
    if pending_docs:
        flushed = flush_pending()
        print(f"\n[最终提交] 写入 {flushed} 个文本块到向量库")

    # 完成总结
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("摄入完成！")
    print("=" * 80)
    print(f"尝试处理: {len(files_to_ingest)} 个文件")
    print(f"成功: {stats['success']} 个文件")
    print(f"失败: {stats['failed']} 个文件")
    print(f"跳过: {stats['skipped']} 个文件")
    print(f"新增文本块: {stats['total_chunks']}")
    print(f"API 错误: {stats['api_errors']} 个 embedding")
    print(f"向量库总文档数: {collection.count()}")
    print(f"耗时: {total_time/60:.1f} 分钟")

    if stats['errors']:
        print(f"\n错误列表（前30个）:")
        for err in stats['errors'][:30]:
            print(f"  - {err}")

    # 关闭 httpx 客户端
    try:
        if _http_client is not None:
            _http_client.close()
    except Exception:
        pass

    # 更新 existing_sources.json 缓存
    print("\n正在更新 existing_sources.json 缓存...")
    try:
        all_new_sources = set(existing_sources)
        for f in files_to_ingest:
            all_new_sources.add(f['filename'])
        import json
        cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "existing_sources.json")
        with open(cache_file, 'w', encoding='utf-8') as fp:
            json.dump(sorted(all_new_sources), fp, ensure_ascii=False)
        print(f"缓存已更新，共 {len(all_new_sources)} 个来源文件")
    except Exception as e:
        print(f"缓存更新失败: {e}")


if __name__ == "__main__":
    main()
