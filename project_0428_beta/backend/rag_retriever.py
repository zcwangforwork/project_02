"""
医疗器械体系文件审核 - RAG 检索模块
实现多轮审核流水线：章节分割 → 逐章并发审核 → 综合分析

全生命周期文档处理：设计开发 + 合规注册 + 生产质控
"""
import os
import asyncio
import json
import logging
import threading
import gc
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import httpx

from audit_prompts import (
    get_section_prompt,
    get_synthesis_prompt,
    get_audit_type_label,
    AUDIT_TYPE_MAP,
)

logger = logging.getLogger(__name__)


class RAGRetriever:
    """RAG 检索器，用于医疗器械体系文件全生命周期审核"""

    # 保留向后兼容的默认 Prompt（当 audit_prompts 不可用时使用）
    FALLBACK_SECTION_PROMPT = """你是一个专业的医疗器械企业体系文件审核专家。
对用户文档的当前章节进行审核，给出差距分析和修改建议。

## 输出格式
### 原文摘要 | ### 标准要求 | ### 差距分析 | ### 修改建议 | ### 关联法规条款 | ### 严重度评级
"""

    FALLBACK_SYNTHESIS_PROMPT = """你是一个专业的医疗器械企业体系文件审核专家。
汇总各章节审核结果，生成最终审核报告。

## 输出格式
# 体系文件审核报告
## 一、审核概述 | ## 二、严重问题汇总 | ## 三、需要修改项汇总 | ## 四、基本符合项 | ## 五、缺失章节 | ## 六、修改优先级建议 | ## 七、关联法规条款总览
"""

    def __init__(self, vector_store, api_key: str, api_url: str, model: str = "glm-5.1"):
        self.vector_store = vector_store
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        # 共享 httpx.AsyncClient 实例，避免并发时重复创建
        self._http_client: Optional[httpx.AsyncClient] = None
        # 线程锁，序列化 ChromaDB HNSW 索引查询，避免多线程同时加载索引导致 OOM
        self._chroma_lock = threading.Lock()

    async def _get_http_client(self) -> httpx.AsyncClient:
        """获取或创建共享的 httpx.AsyncClient"""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=600, trust_env=False)
        return self._http_client

    async def close(self):
        """关闭共享的 httpx 客户端"""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    def _retrieve_for_section_sync(self, section_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        针对特定段落检索相关的知识库内容（同步方法，供 run_in_executor 调用）

        使用线程锁序列化 ChromaDB 查询，避免多线程同时加载 HNSW 索引导致 OOM

        Args:
            section_text: 段落文本
            n_results: 返回结果数量

        Returns:
            相关文档片段列表
        """
        # 序列化 ChromaDB 查询，防止多个线程同时加载 HNSW 索引
        with self._chroma_lock:
            results = self.vector_store.query(
                query_texts=[section_text],
                n_results=n_results * 2
            )

        docs = []
        if results and 'documents' in results:
            for i, doc_text in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if 'metadatas' in results else {}
                source = metadata.get("source", "unknown")
                # 优先保留风险管理相关文档
                if any(kw in source.lower() for kw in ['风险', 'risk', '14971', '42062', '9706']):
                    docs.append({
                        "text": doc_text,
                        "source": source,
                        "chunk_id": metadata.get("chunk_id", i)
                    })
                    if len(docs) >= n_results:
                        break

        # 补充其他文档
        if len(docs) < n_results:
            for i, doc_text in enumerate(results['documents'][0]):
                if len(docs) >= n_results:
                    break
                metadata = results['metadatas'][0][i] if 'metadatas' in results else {}
                source = metadata.get("source", "unknown")
                if not any(d['source'] == source for d in docs):
                    docs.append({
                        "text": doc_text,
                        "source": source,
                        "chunk_id": metadata.get("chunk_id", i)
                    })

        return docs

    async def _retrieve_for_section(self, section_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        异步检索知识库内容（包装同步 ChromaDB 调用，避免阻塞事件循环）

        Args:
            section_text: 段落文本
            n_results: 返回结果数量

        Returns:
            相关文档片段列表
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._retrieve_for_section_sync,
            section_text,
            n_results
        )

    async def _call_llm(self, system_prompt: str, user_content: str, max_tokens: int = 4000, temperature: float = 0.7) -> str:
        """
        调用 LLM API，带重试机制

        Args:
            system_prompt: 系统提示词
            user_content: 用户内容
            max_tokens: 最大输出 token 数
            temperature: 温度参数

        Returns:
            LLM 回答文本
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        client = await self._get_http_client()

        # 重试逻辑：最多3次，指数退避
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await client.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                # 安全解析 JSON：API 可能返回非 JSON 错误文本
                try:
                    result = response.json()
                except json.JSONDecodeError:
                    response_text = response.text[:500]
                    logger.warning(f"API 返回非 JSON 响应: {response_text}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                        continue
                    raise RuntimeError(f"API 返回了非 JSON 格式的响应: {response_text}")

                answer = ""
                choices = result.get("choices", [])
                if choices and len(choices) > 0:
                    choice = choices[0]
                    message = choice.get("message", {})
                    answer = message.get("content", "")

                return answer or str(result)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # 速率限制，指数退避
                    wait_time = 2 ** attempt
                    logger.warning(f"API 速率限制，等待 {wait_time} 秒后重试 (尝试 {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                raise
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"API 请求失败: {e}，等待 {wait_time} 秒后重试")
                    await asyncio.sleep(wait_time)
                    continue
                raise

        return ""

    async def _audit_single_section(
        self,
        section_title: str,
        section_content: str,
        section_level: int,
        audit_type: str,
        semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """
        审核单个章节

        Args:
            section_title: 章节标题
            section_content: 章节内容
            section_level: 标题层级
            audit_type: 审核类型
            semaphore: 并发信号量

        Returns:
            包含审核结果的字典
        """
        async with semaphore:
            try:
                # 检索相关知识库内容
                relevant_docs = await self._retrieve_for_section(section_content, n_results=5)

                # 根据审核类型选择对应的专用 System Prompt（从 audit_prompts 模块加载）
                try:
                    system_prompt = get_section_prompt(audit_type)
                    audit_label = get_audit_type_label(audit_type)
                except Exception:
                    system_prompt = self.FALLBACK_SECTION_PROMPT
                    audit_label = "体系文件审核"

                # 构建每章节的上下文
                context_parts = [
                    f"## 审核类型：{audit_label}",
                    f"## 当前审核章节：{section_title}\n",
                    "### 用户文档原文：",
                    section_content[:3000],
                    "\n### 标准要求/参考模板（来自知识库）："
                ]

                if relevant_docs:
                    for i, doc in enumerate(relevant_docs, 1):
                        context_parts.append(f"\n--- 参考 {i} (来源: {doc['source']}) ---")
                        context_parts.append(doc['text'][:2000])
                else:
                    context_parts.append("[未在知识库中找到直接相关的参考内容，请基于标准原文进行审核]")

                user_content = '\n'.join(context_parts)

                # 调用 LLM
                answer = await self._call_llm(
                    system_prompt=system_prompt,
                    user_content=user_content,
                    max_tokens=4000,
                    temperature=0.7
                )

                # 提取摘要（用于综合分析）
                summary = self._extract_section_summary(answer)

                return {
                    "title": section_title,
                    "level": section_level,
                    "answer": answer,
                    "summary": summary,
                    "relevant_docs": relevant_docs,
                    "status": "success"
                }
            except Exception as e:
                logger.error(f"章节 '{section_title}' 审核失败: {e}")
                return {
                    "title": section_title,
                    "level": section_level,
                    "answer": f"### 审核失败\n\n该章节审核过程中发生错误: {str(e)}",
                    "summary": f"审核失败: {str(e)}",
                    "relevant_docs": [],
                    "status": "error"
                }

    def _extract_section_summary(self, answer: str) -> str:
        """
        从章节审核结果中提取摘要（用于综合分析阶段）

        提取严重度评级和关键发现，限制在200字以内
        """
        lines = answer.split('\n')
        summary_parts = []
        for line in lines:
            stripped = line.strip()
            # 提取严重度评级
            if '🔴' in stripped or '🟡' in stripped or '🟢' in stripped:
                summary_parts.append(stripped)
            # 提取差距分析标题行
            elif stripped.startswith('- ') and len(summary_parts) < 5:
                summary_parts.append(stripped)

        if not summary_parts:
            # 降级：取前200字
            return answer[:200].strip()

        return '\n'.join(summary_parts[:8])

    async def analyze_document(
        self,
        user_document: str,
        user_filename: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        use_contrast_mode: bool = True,
        audit_type: str = "risk_management"
    ) -> Dict[str, Any]:
        """
        使用多轮审核流水线分析用户文档

        流水线：
        1. 代码分割：按 Markdown 标题切分章节
        2. 逐章并发审核：每章节独立 LLM 调用，asyncio.gather + Semaphore(2)
        3. 综合分析：汇总各章节摘要，生成最终报告

        Args:
            user_document: 用户文档内容（Markdown 格式）
            user_filename: 用户文件名
            temperature: 温度参数
            max_tokens: 每章节最大输出 token 数
            use_contrast_mode: 是否使用对照审核模式
            audit_type: 审核类型

        Returns:
            包含完整审核报告的字典
        """
        from doc_processor import split_by_markdown_headers

        # ===== 第一轮：代码分割章节 =====
        sections = split_by_markdown_headers(user_document)
        logger.info(f"文档分割为 {len(sections)} 个章节")

        if not sections:
            # 降级到原始模式
            return await self._analyze_document_fallback(user_document, user_filename, audit_type)

        # 限制最多审核 20 个章节，避免超长文档导致内存溢出
        MAX_SECTIONS = 20
        if len(sections) > MAX_SECTIONS:
            logger.warning(f"文档章节过多({len(sections)})，仅审核前 {MAX_SECTIONS} 个")
            sections = sections[:MAX_SECTIONS]

        # ===== 第二轮：逐章并发审核 =====
        # 限制并发数为 2，避免 ChromaDB HNSW 索引多线程同时加载导致内存溢出
        semaphore = asyncio.Semaphore(2)

        tasks = []
        for title, content, level in sections:
            tasks.append(
                self._audit_single_section(title, content, level, audit_type, semaphore)
            )

        section_results = await asyncio.gather(*tasks)

        # 释放原始文档文本（已分割为章节，不再需要全文）
        del user_document
        gc.collect()

        # 汇总所有检索到的文档（合并去重）
        seen_sources = set()
        all_retrieved_docs = []
        for result in section_results:
            for doc in result.get("relevant_docs", []):
                source = doc.get("source", "")
                if source not in seen_sources:
                    seen_sources.add(source)
                    all_retrieved_docs.append(doc)
            # 清除 relevant_docs 以释放内存（后续只用 summary 和 answer）
            result.pop("relevant_docs", None)

        # ===== 第三轮：综合分析 =====
        try:
            # 获取审核类型标签
            try:
                audit_label = get_audit_type_label(audit_type)
            except Exception:
                audit_label = "综合体系审核"

            # 构建综合分析上下文（只传摘要，不传全文）
            synthesis_parts = [
                f"审核类型：{audit_label}",
                f"以下是文档「{user_filename}」各章节的审核摘要：\n"
            ]

            for result in section_results:
                level_prefix = "#" * max(result["level"], 1) if result["level"] > 0 else "##"
                synthesis_parts.append(f"{level_prefix} {result['title']}")
                synthesis_parts.append(result['summary'])
                synthesis_parts.append("")

            # 添加文档章节列表（用于缺失章节检测）
            synthesis_parts.append("\n---\n文档完整章节列表：")
            for result in section_results:
                indent = "  " * (result["level"] - 1) if result["level"] > 0 else ""
                synthesis_parts.append(f"{indent}- {result['title']}")

            synthesis_context = '\n'.join(synthesis_parts)

            # 选择对应阶段的综合分析 Prompt
            try:
                synthesis_prompt = get_synthesis_prompt(audit_type)
            except Exception:
                synthesis_prompt = self.FALLBACK_SYNTHESIS_PROMPT

            synthesis_answer = await self._call_llm(
                system_prompt=synthesis_prompt,
                user_content=synthesis_context,
                max_tokens=8000,
                temperature=0.5
            )
        except Exception as e:
            logger.error(f"综合分析失败: {e}")
            synthesis_answer = "## 综合分析\n\n综合分析阶段发生错误，请参考各章节独立审核结果。"

        # ===== 组装最终报告 =====
        # 内存优化：限制最终报告大小，防止前端浏览器因 DOM 过大而崩溃
        MAX_FINAL_ANSWER_CHARS = 30000  # 最终报告最大字符数
        MAX_SECTION_ANSWER_CHARS = 2000  # 每个章节详情在最终报告中的最大字符数

        final_parts = [synthesis_answer, "\n\n---\n\n# 各章节详细审核结果\n"]

        for i, result in enumerate(section_results, 1):
            level_prefix = "#" * max(result["level"], 1) if result["level"] > 0 else "##"
            final_parts.append(f"\n{level_prefix} 审核点 {i}：{result['title']}\n")
            # 截断过长的章节回答
            section_answer = result['answer']
            if len(section_answer) > MAX_SECTION_ANSWER_CHARS:
                section_answer = section_answer[:MAX_SECTION_ANSWER_CHARS] + f"\n\n... (该章节审核结果共 {len(result['answer'])} 字符，已截断。完整结果请参见上方综合分析)"
            final_parts.append(section_answer)
            final_parts.append("\n---\n")

        final_answer = '\n'.join(final_parts)

        # 内存保护：最终报告超出上限时进一步截断
        if len(final_answer) > MAX_FINAL_ANSWER_CHARS:
            logger.warning(f"最终报告过大({len(final_answer)}字符)，截断至{MAX_FINAL_ANSWER_CHARS}字符")
            final_answer = final_answer[:MAX_FINAL_ANSWER_CHARS] + (
                f"\n\n---\n\n⚠️ 审核报告过长，共 {len(final_answer)} 字符，"
                f"已截断至 {MAX_FINAL_ANSWER_CHARS} 字符。"
                f"共审核了 {len(sections)} 个章节，详细结果请查看上方各章节分析。"
            )

        result = {
            "answer": final_answer,
            "section_count": len(sections),
            "section_results": [
                {"title": r["title"], "status": r["status"]}
                for r in section_results
            ],
            "retrieved_docs": all_retrieved_docs
        }

        # 显式释放大对象以帮助 GC
        final_parts.clear()
        synthesis_parts.clear()
        for r in section_results:
            r.clear()
        section_results.clear()
        gc.collect()

        return result

    async def _analyze_document_fallback(
        self,
        user_document: str,
        user_filename: str,
        audit_type: str
    ) -> Dict[str, Any]:
        """
        降级模式：当文档无法分割章节时，使用单次调用审核

        Args:
            user_document: 用户文档内容
            user_filename: 用户文件名
            audit_type: 审核类型

        Returns:
            审核结果
        """
        try:
            system_prompt = get_section_prompt(audit_type)
        except Exception:
            system_prompt = self.FALLBACK_SECTION_PROMPT

        # 简单检索
        relevant_docs = await self._retrieve_for_section(user_document, n_results=5)

        context_parts = [
            f"文档名: {user_filename}\n",
            "### 用户文档原文：",
            user_document[:8000],
            "\n### 标准要求/参考模板（来自知识库）："
        ]
        for i, doc in enumerate(relevant_docs, 1):
            context_parts.append(f"\n--- 参考 {i} (来源: {doc['source']}) ---")
            context_parts.append(doc['text'][:2000])

        user_content = '\n'.join(context_parts)

        answer = await self._call_llm(
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=8000,
            temperature=0.7
        )

        return {
            "answer": answer,
            "section_count": 1,
            "section_results": [{"title": "完整文档", "status": "success"}],
            "retrieved_docs": relevant_docs
        }

    # ============== 兼容旧接口 ==============
    def retrieve_relevant_docs(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """兼容旧接口：从知识库检索相关片段"""
        return self._retrieve_for_section_sync(query_text, n_results=n_results)

    def build_context(self, user_document: str, user_filename: str, n_results: int = 5, use_contrast_mode: bool = True) -> Tuple[str, List[Dict[str, Any]]]:
        """兼容旧接口：构建增强上下文"""
        relevant_docs = self._retrieve_for_section_sync(user_document, n_results=n_results)
        context_parts = [
            "=" * 60,
            "【用户上传的体系文件】",
            f"文件名: {user_filename}",
            "=" * 60,
            user_document[:8000],
            "\n",
            "=" * 60,
            "【知识库检索结果 - 相关法规和模板参考】",
            "=" * 60
        ]
        for i, doc in enumerate(relevant_docs, 1):
            context_parts.append(f"\n--- 参考文档 {i} (来源: {doc['source']}) ---")
            context_parts.append(doc['text'][:2000])
        return '\n'.join(context_parts), relevant_docs


def create_rag_retriever(vector_store, api_key: str, api_url: str, model: str = "glm-5.1") -> RAGRetriever:
    """
    工厂函数：创建 RAG 检索器实例

    Args:
        vector_store: VectorStore 实例
        api_key: API 密钥
        api_url: API 地址
        model: 模型名称

    Returns:
        RAGRetriever 实例
    """
    return RAGRetriever(
        vector_store=vector_store,
        api_key=api_key,
        api_url=api_url,
        model=model
    )
