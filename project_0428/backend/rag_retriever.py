"""
医疗器械体系文件审核 - RAG 检索模块
实现检索增强生成，组合用户文档和知识库进行审核
"""
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import httpx


class RAGRetriever:
    """RAG 检索器，用于医疗器械体系文件审核"""

    # 风险管理专项审核的增强 System Prompt（用于对照审核模式）
    RISK_MANAGEMENT_CONTRAST_PROMPT = """你是一个专业的医疗器械企业体系文件审核专家，精通ISO 14971:2019风险管理标准和风险管理文档的编写规范。

你的任务是对照知识库中的标准模板和参考文档，对用户上传的风险管理文档进行**逐条对照审核**，指出每一处与标准要求的差距，并给出具体的修改建议。

## 核心审核原则

### 1. 逐条对照审核
对用户文档的**每一个章节/段落**，都要：
1. 明确指出该段落对应知识库中的哪些参考内容
2. 详细分析当前内容与标准要求的差距
3. 给出具体的修改建议

### 2. 风险管理文档结构要求（ISO 14971:2019）
标准风险管理文档应包含以下章节：
- **风险管理计划**：范围界定、职责分配、阶段评审点、风险可接受准则
- **风险管理范围**：预期用途、合理可预见的误用、适用范围（适应症/禁忌症）
- **危害识别**：所有可能的危害源（物理危害、化学危害、生物危害、功能危害等）
- **危害处境和损害情形**：描述完整的危害链（危害→危害处境→损害）
- **风险评估**：概率估计、严重度评估、风险等级判定
- **风险控制**：控制措施、控制措施验证、剩余风险评估
- **综合剩余风险**：总体剩余风险是否可接受
- **风险管理评审**：生产和生产后活动与风险管理计划的符合性
- **生产和生产后信息**：上市后监督、不良事件、召回信息

### 3. 对照输出格式
对每个审核点，请严格按以下格式输出：

---

## 【审核点 N】章节标题

### 原文内容
[摘录用户文档中该章节的原文]

### 标准要求（对照参考）
[列出知识库中对应章节的要求内容，注明来源文件名]

### 差距分析
[详细分析用户文档内容与标准要求之间的具体差距]

### 修改建议
[给出具体的、可操作的修改建议，最好有示例表述]

### 关联法规条款
[引用ISO 14971:2019的具体条款号，如4.1、5.2、5.5等]

---

## 附加审核要点

### 风险可接受准则
- 是否明确定义了风险可接受准则
- 准则是否与产品特点相适应
- 是否区分了单个风险和综合剩余风险的接受标准

### 文档一致性
- 文档内部的术语、定义是否一致
- 各章节之间是否相互呼应、无矛盾
- 风险控制措施是否与危害识别相对应

### 证据支撑
- 是否有足够的技术资料支撑风险估计
- 风险控制措施的验证证据是否充分

## 重要提示
1. **必须**对照知识库中的具体参考文档，引用时要写明来源文件名
2. 修改建议要具体到可以指导用户直接修改的程度
3. 如果发现文档缺少整个章节，必须指出并给出该章节的编写指导
4. 如果知识库中没有相关的详细参考，请在"标准要求"中注明"建议参考ISO 14971:2019第X条原文"
5. 保持专业、严谨的语言风格，使用医疗器械行业的标准术语
"""

    # 原有通用审核的 System Prompt（保留兼容）
    GENERAL_AUDIT_PROMPT = """你是一个专业的医疗器械企业体系文件审核专家。

你的任务是审核用户上传的体系文件，并结合知识库中的标准文档给出修改建议。

## 审核范围
- ISO 13485:2016 医疗器械质量管理体系
- ISO 14971:2019 医疗器械风险管理的应用
- IEC 62304 医疗器械软件生命周期过程
- EU MDR 2017/745 欧盟医疗器械法规
- NMPA 医疗器械生产质量管理规范
- FDA 21 CFR Part 820 质量体系法规

## 审核原则
1. **完整性检查**：文件是否覆盖相关法规条款的要求
2. **一致性检查**：文件内容是否相互协调、无矛盾
3. **可操作性**：文件描述是否足够具体、可执行
4. **证据链**：是否有相应的记录表单支撑执行证据

## 输出格式
请按以下格式输出审核结果：

### 一、缺失内容清单
（列出文件中完全没有涉及的重要条款或要求）

### 二、需要修改的地方
（列出内容不完整、不准确或需要强化的条款）

### 三、具体修改建议
（对每个问题给出具体的修改方向）

### 四、关联法规条款依据
（引用相关的标准条款作为修改依据）

## 重要提示
- 直接给出审核结果，不要输出思考过程
- 如果知识库中没有检索到相关内容，请在"关联法规条款依据"中注明"该部分建议参考对应标准原文"
- 回答应专业、具体、可操作
"""

    SYSTEM_PROMPT = GENERAL_AUDIT_PROMPT

    def __init__(self, vector_store, api_key: str, api_url: str, model: str = "MiniMax-M2.7"):
        """
        初始化 RAG 检索器

        Args:
            vector_store: VectorStore 实例
            api_key: API 密钥
            api_url: API 地址
            model: 模型名称
        """
        self.vector_store = vector_store
        self.api_key = api_key
        self.api_url = api_url
        self.model = model

    def _split_document_into_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        将文档分割成逻辑段落，每个段落包含(标题, 内容)

        Args:
            text: 文档全文

        Returns:
            [(标题, 内容), ...] 段落列表
        """
        sections = []
        lines = text.split('\n')
        current_title = "文档开头"
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测可能是标题的行（短行+特定开头/或数字编号）
            is_title = False
            if len(line) < 50:
                # 各种标题模式
                title_patterns = [
                    '1.', '2.', '3.', '4.', '5.',  # 数字编号
                    '一、', '二、', '三、', '四、', '五、', '六、', '七、', '八、', '九、', '十、',  # 中文编号
                    '（一）', '（二）', '（三）', '（四）', '（五）',  # 中文括号
                    '第1章', '第2章', '第3章', '第1节', '第2节',
                    '风险管理', '危害识别', '风险评估', '风险控制', '风险总结', '综合剩余风险',
                    '文档', '范围', '目的', '适用', '定义', '引用', '参考', '附录'
                ]
                for pattern in title_patterns:
                    if line.startswith(pattern):
                        is_title = True
                        break

            if is_title and current_content:
                # 保存上一个段落
                full_content = '\n'.join(current_content)
                if len(full_content) > 50:  # 只保存有实质内容的段落
                    sections.append((current_title, full_content))
                current_title = line
                current_content = []
            else:
                current_content.append(line)

        # 保存最后一个段落
        if current_content:
            full_content = '\n'.join(current_content)
            if len(full_content) > 50:
                sections.append((current_title, full_content))

        return sections

    def _retrieve_for_section(self, section_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        针对特定段落检索相关的知识库内容

        Args:
            section_text: 段落文本
            n_results: 返回结果数量

        Returns:
            相关文档片段列表
        """
        # 使用段落文本作为查询，检索相关知识库内容
        results = self.vector_store.query(
            query_texts=[section_text],
            n_results=n_results * 2  # 多检索一些，过滤后取前n_results个
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

        # 如果没有找到足够的风控相关文档，补充其他文档
        if len(docs) < n_results:
            for i, doc_text in enumerate(results['documents'][0]):
                if len(docs) >= n_results:
                    break
                metadata = results['metadatas'][0][i] if 'metadatas' in results else {}
                source = metadata.get("source", "unknown")
                # 避免重复
                if not any(d['source'] == source for d in docs):
                    docs.append({
                        "text": doc_text,
                        "source": source,
                        "chunk_id": metadata.get("chunk_id", i)
                    })

        return docs

    def retrieve_relevant_docs(
        self,
        query_text: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        从知识库检索与用户文档相关的片段

        Args:
            query_text: 用户文档文本
            n_results: 返回结果数量

        Returns:
            相关文档片段列表
        """
        results = self.vector_store.query(
            query_texts=[query_text],
            n_results=n_results * 2  # 多检索一些，过滤后取前n_results个
        )

        docs = []
        if results and 'documents' in results:
            for i, doc_text in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if 'metadatas' in results else {}
                source = metadata.get("source", "unknown")
                # 过滤掉FDA相关的文档
                if 'FDA' in source or 'fda' in source.lower():
                    continue
                docs.append({
                    "text": doc_text,
                    "source": source,
                    "chunk_id": metadata.get("chunk_id", i)
                })
                if len(docs) >= n_results:
                    break

        return docs

    def build_context(
        self,
        user_document: str,
        user_filename: str,
        n_results: int = 5,
        use_contrast_mode: bool = True
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        构建增强上下文，组合用户文档和检索结果

        Args:
            user_document: 用户文档内容
            user_filename: 用户文件名
            n_results: 检索结果数量
            use_contrast_mode: 是否使用分段对照模式（默认True）

        Returns:
            (增强后的上下文字符串, 检索到的相关文档列表)
        """
        if use_contrast_mode:
            return self._build_contrast_context(user_document, user_filename, n_results)
        else:
            return self._build_standard_context(user_document, user_filename, n_results)

    def _build_standard_context(
        self,
        user_document: str,
        user_filename: str,
        n_results: int = 5
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        构建标准上下文（简单拼接模式）

        Args:
            user_document: 用户文档内容
            user_filename: 用户文件名
            n_results: 检索结果数量

        Returns:
            (增强后的上下文字符串, 检索到的相关文档列表)
        """
        # 检索相关知识库内容
        relevant_docs = self.retrieve_relevant_docs(user_document, n_results=n_results)

        # 构建上下文
        context_parts = [
            "=" * 60,
            "【用户上传的体系文件】",
            f"文件名: {user_filename}",
            "=" * 60,
            user_document[:8000],  # 限制用户文档长度
            "\n",
            "=" * 60,
            "【知识库检索结果 - 相关法规和模板参考】",
            "=" * 60
        ]

        for i, doc in enumerate(relevant_docs, 1):
            context_parts.append(f"\n--- 参考文档 {i} (来源: {doc['source']}) ---")
            context_parts.append(doc['text'][:2000])  # 限制每个片段长度

        return '\n'.join(context_parts), relevant_docs

    def _build_contrast_context(
        self,
        user_document: str,
        user_filename: str,
        n_results: int = 3
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        构建分段对照上下文 - 每个文档段落对应知识库参考内容

        Args:
            user_document: 用户文档内容
            user_filename: 用户文件名
            n_results: 每个段落检索的结果数量

        Returns:
            (分段对照的上下文字符串, 所有检索到的相关文档列表)
        """
        # 将文档分割成段落
        sections = self._split_document_into_sections(user_document)

        # 限制段落数量，最多处理前8个段落以控制上下文长度
        max_sections = 8
        if len(sections) > max_sections:
            sections = sections[:max_sections]

        context_parts = [
            "=" * 70,
            f"【用户上传的风险管理文档】文件名: {user_filename}",
            f"【文档已自动分割为 {len(sections)} 个章节进行对照审核】",
            "=" * 70,
            ""
        ]

        all_retrieved_docs = []

        for idx, (title, content) in enumerate(sections, 1):
            context_parts.append("")
            context_parts.append("=" * 70)
            context_parts.append(f"## 【审核点 {idx}】{title}")
            context_parts.append("=" * 70)

            # 用户文档该段落的内容
            context_parts.append("")
            context_parts.append("### 用户文档原文：")
            context_parts.append(content[:1500])  # 限制每段长度

            # 针对该段落检索相关知识库内容
            relevant_for_section = self._retrieve_for_section(content, n_results=n_results)
            all_retrieved_docs.extend(relevant_for_section)

            if relevant_for_section:
                context_parts.append("")
                context_parts.append("### 标准要求/参考模板（来自知识库）：")
                for i, doc in enumerate(relevant_for_section, 1):
                    context_parts.append(f"\n--- 参考 {i} (来源: {doc['source']}) ---")
                    context_parts.append(doc['text'][:1500])  # 限制每个片段长度
            else:
                context_parts.append("\n### 标准要求：")
                context_parts.append("[未在知识库中找到直接相关的参考内容，建议查阅ISO 14971:2019对应条款原文]")

            context_parts.append("")
            context_parts.append("### 差距分析与修改建议：")
            context_parts.append("[请在下方空白处填写对该章节的详细审核意见，包括与标准要求的差距和具体修改建议]")

            context_parts.append("")

        # 添加综合审核说明
        context_parts.append("")
        context_parts.append("=" * 70)
        context_parts.append("## 【综合审核说明】")
        context_parts.append("=" * 70)
        context_parts.append("""
请按照上述分段结构，对每个审核点进行详细分析。输出格式要求：
1. 对每个审核点，先引用用户文档原文，再引用知识库参考内容
2. 明确指出差距（用户文档少了什么、错了什么、不准确什么）
3. 给出具体的修改建议和示例表述
4. 引用相关的ISO 14971:2019条款号作为依据

审核重点关注：
- 风险管理计划的完整性和可操作性
- 危害识别的全面性（是否漏掉了重要危害源）
- 风险评估的合理性（概率和严重度估计是否有依据）
- 风险控制措施的具体性和有效性
- 剩余风险评估和综合剩余风险判定
- 文档各章节之间的一致性和逻辑连贯性
""")

        return '\n'.join(context_parts), all_retrieved_docs

    async def analyze_document(
        self,
        user_document: str,
        user_filename: str,
        temperature: float = 0.7,
        max_tokens: int = 32000,
        use_contrast_mode: bool = True,
        audit_type: str = "risk_management"
    ) -> Dict[str, Any]:
        """
        使用 RAG 增强的方式分析用户文档

        Args:
            user_document: 用户文档内容
            user_filename: 用户文件名
            temperature: 温度参数
            max_tokens: 最大令牌数
            use_contrast_mode: 是否使用分段对照模式（默认True）
            audit_type: 审核类型，"risk_management"使用风险管理专项prompt，"general"使用通用prompt

        Returns:
            包含回答和引用信息的字典
        """
        # 构建增强上下文（使用分段对照模式）
        context, relevant_docs = self.build_context(
            user_document,
            user_filename,
            n_results=3,
            use_contrast_mode=use_contrast_mode
        )

        # 根据审核类型选择System Prompt
        if audit_type == "risk_management" and use_contrast_mode:
            system_prompt = self.RISK_MANAGEMENT_CONTRAST_PROMPT
        else:
            system_prompt = self.SYSTEM_PROMPT

        # 构建完整的消息
        if use_contrast_mode and audit_type == "risk_management":
            full_content = f"""请对以下风险管理文档进行**逐条对照审核**：

{context}

---

请严格按照上述分段结构，对每个审核点进行详细分析。
对每个审核点：
1. 先引用用户文档原文
2. 再引用知识库参考内容
3. 明确指出差距
4. 给出具体修改建议
5. 引用ISO 14971:2019相关条款号

请开始审核："""
        else:
            full_content = f"""请审核以下医疗器械体系文件，结合知识库给出修改建议：

{context}

---

请根据上述文件内容，按照输出格式进行审核："""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_content}
        ]

        # 调用 API (OpenAI 兼容格式)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        async with httpx.AsyncClient(timeout=600, trust_env=False) as client:
            response = await client.post(
                self.api_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()

        # 解析回答 - OpenAI 兼容格式
        answer = ""
        choices = result.get("choices", [])
        if choices and len(choices) > 0:
            choice = choices[0]
            message = choice.get("message", {})
            answer = message.get("content", "")

        if not answer:
            answer = str(result) if result else ""

        return {
            "answer": answer,
            "usage": result.get("usage"),
            "retrieved_docs": relevant_docs
        }


def create_rag_retriever(vector_store, api_key: str, api_url: str, model: str = "MiniMax-M2.7") -> RAGRetriever:
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
