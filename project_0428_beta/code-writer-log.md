# Code Writer Log

## 2026-05-29 14:30 - 全生命周期文档处理模式扩展

### Overview
扩展文档上传后的处理方式，从原来的2种审核模式（风险管理、通用）扩展为覆盖医疗器械产品全生命周期的27种文档处理模式，涵盖设计开发、风险管理、合规注册、生产质控四大阶段。

### New File Created
- `backend/audit_prompts.py` (913 lines): 全生命周期审核提示词模块
  - 5组逐章审核System Prompt (设计开发/风险管理/合规注册/生产质控/通用)
  - 5组综合分析System Prompt
  - 27种审核类型映射表 (AUDIT_TYPE_MAP)
  - 审计类型中文标签 (AUDIT_TYPE_LABELS)
  - 生命周期阶段分组 (LIFECYCLE_PHASES)
  - 提示词获取函数 (get_section_prompt/get_synthesis_prompt/get_audit_type_label)

### Files Modified
- `backend/rag_retriever.py` (from 647 to 575 lines):
  - 移除内联的3个旧Prompt常量
  - 导入audit_prompts模块的提示词获取函数
  - 更新_audit_single_section()使用get_section_prompt()
  - 更新_analyze_document_fallback()使用新提示词
  - 更新analyze_document()综合分析使用get_synthesis_prompt()
  - 在用户上下文中添加审核类型标签
  - 保留FALLBACK_PROMPT以兼容模块导入失败的情况

- `backend/main.py` (from 851 to 908 lines):
  - 导入audit_prompts模块
  - 扩展/api/analyze的audit_type验证，支持全部27种类型
  - 新增/api/audit-types端点，向前端提供生命周期阶段和审核类型信息
  - 更新/info端点返回audit_types和lifecycle_phases
  - 更新/api/analyze docstring

- `frontend/index.html` (from 911 to 1035 lines):
  - 重新设计上传模态框处理方式区域
  - 新增6个生命周期阶段标签页（设计开发/风险管理/合规注册/生产质控/综合审核/其他）
  - 每个阶段有独立文档类型下拉列表
  - 动态审核类型说明更新
  - 阶段切换功能和颜色编码

### Design & Development types (10):
design_planning, design_input, design_input_review, design_output,
design_review, design_verification, design_validation, design_change,
design_transfer, dhf

### Compliance & Registration types (6):
regulatory_submission, technical_documentation, clinical_evaluation,
labeling_ifu, standards_compliance, essential_requirements

### Production & QC types (9):
sop, batch_record, process_validation, qc_testing, capa,
supplier_management, equipment_management, environment_monitoring, traceability

### Verification
- Python syntax checks: all 3 files OK
- Existing unit tests: 10/10 passed
- Integration routing test: all 27 audit types correctly routed
- Prompt sizes verified (design_development: 1997+974 chars, risk_management: 2251+943 chars)

## 2026-05-19 - Memory Optimization: 文件上传分析 OOM 修复

### Files Modified
- `backend/vector_store.py`: ChromaDB 内存优化
- `backend/rag_retriever.py`: 并发控制和内存清理

### Root Cause
ChromaDB 数据库 `chroma.sqlite3` 达 2.9GB，HNSW 索引在多线程并发查询时被多次加载到内存中，导致 OOM。

### Fixes Applied
1. **vector_store.py**: HNSW 参数优化 (M=8, search_ef=50), 限制 n_results≤10
2. **rag_retriever.py**: 
   - 添加 threading.Lock 序列化 ChromaDB 查询
   - Semaphore 5→2
   - MAX_SECTIONS=20 限制
   - del + gc.collect() 释放文档全文
   - 清除中间结果 relevant_docs

## 2026-04-28 - Project Created: 金融问答助手

### Task: 创建金融问答助手 Web 应用

#### Project Path
`E:\nrf_sample_codes\project_0428`

#### Files Created

1. **backend/main.py** - FastAPI 后端主程序
   - 实现了 `/`, `/health`, `/api/chat`, `/api/clear`, `/api/history/{session_id}` 等 API 端点
   - 配置管理通过环境变量 (OPENAI_API_KEY, OPENAI_API_URL, OPENAI_MODEL, REQUEST_TIMEOUT)
   - 包含金融领域 system prompt
   - 支持多轮对话上下文管理
   - CORS 中间件配置

2. **backend/requirements.txt** - Python 依赖
   - fastapi>=0.100.0
   - uvicorn[standard]>=0.23.0
   - httpx>=0.24.0
   - pydantic>=2.0.0
   - python-dotenv>=1.0.0

3. **frontend/index.html** - 前端页面
   - 使用 TailwindCSS CDN 实现现代化 UI
   - 响应式设计，深色主题
   - 消息输入、发送、清空对话功能
   - 加载动画、错误处理
   - API Key 配置提示

4. **README.md** - 项目文档
   - 功能特性说明
   - 快速开始指南
   - API 接口文档
   - 配置说明

#### Verification
- Python syntax check: Passed

## 2026-04-29 10:30:00 - Enhancement: 风险管理对照审核功能

### Task: 增强project_0428的风险管理审核，实现知识库内容逐条对照

#### Project Path
`E:\nrf_sample_codes\code_writer\project_0428`

#### Changes Made

**1. backend/rag_retriever.py**
- 新增 `RISK_MANAGEMENT_CONTRAST_PROMPT` 专门用于风险管理专项对照审核的System Prompt
- 新增 `_split_document_into_sections()` 方法：将用户文档自动分割为逻辑段落
- 新增 `_retrieve_for_section()` 方法：针对每个段落检索相关知识库内容
- 新增 `_build_contrast_context()` 方法：构建分段对照上下文（每段包含原文+参考+审核空白）
- 新增 `_build_standard_context()` 方法：保留原有简单拼接模式（用于general审核）
- 修改 `build_context()` 方法：增加 `use_contrast_mode` 参数
- 修改 `analyze_document()` 方法：增加 `use_contrast_mode` 和 `audit_type` 参数

**2. backend/main.py**
- 修改 `/api/analyze` 端点：增加 `audit_type` 参数（risk_management/general）
- 根据 `audit_type` 决定是否使用分段对照模式

**3. frontend/index.html**
- 处理方式选择增加"风险管理专项审核（对照审核）"和"综合体系文件审核"两个选项
- 新增审核类型说明区域，根据选择动态显示不同提示
- 修改 `confirmUpload()` 函数：正确处理新的审核类型选项并传递给后端API

#### Key Features Added
1. **分段对照审核**：将用户文档分割成多个段落，每个段落对应知识库检索结果
2. **风险管理专项Prompt**：针对ISO 14971:2019的19个条款设计详细审核要点
3. **逐条修改建议**：明确指出用户文档与标准要求的差距，给出具体修改建议
4. **来源标注**：每处参考都标注来自知识库中的哪个文件

#### Verification
- rag_retriever.py syntax check: Passed

## 2026-05-20 17:50 - Browser Memory Crash Fix (OOM)

### Root Cause Analysis
浏览器崩溃的根本原因是三层内存问题叠加：
1. **前端 DOM 无限增长**: 所有聊天消息永久追加到 DOM 中，永不清理
2. **后端返回超大响应**: `analyze_document` 将 20 个章节 × ~4000 tokens 的审核结果拼接成一个巨大字符串返回
3. **前端全量渲染**: 整个响应被 marked.js 解析为 HTML 后全量插入 DOM

### Files Modified

#### 1. `frontend/index.html` - 前端内存优化
- 添加 `MAX_MESSAGES = 30` 常量：DOM 中最多保留 30 个消息气泡
- 添加 `MAX_CONTENT_LENGTH = 3000` 常量：单条消息超过 3000 字符自动折叠
- 新增 `trimOldMessages()` 函数：每次添加消息后检查并移除超出限制的旧消息
- 新增 `createTruncatedContent()` 函数：对超长内容自动折叠，提供"展开/收起"按钮
- 新增 `toggleContent()` 函数：展开/收起切换逻辑
- 优化 `addMessage()`：集成截断渲染和自动裁剪
- 优化 `splitAndAddMessages()`：使用临时容器批量添加消息，统一裁剪
- 新增 `addMessageToFragment()`：创建消息 DOM 节点供批量添加使用
- 添加 `.hidden` 和 `.expand-toggle` CSS 样式

#### 2. `backend/rag_retriever.py` - 后端响应大小限制
- 在 `analyze_document()` 最终报告组装部分添加两层截断：
  - `MAX_SECTION_ANSWER_CHARS = 3000`：每个章节详情截断至 3000 字符
  - `MAX_FINAL_ANSWER_CHARS = 50000`：最终报告总大小截断至 50000 字符
- 添加内存显式释放：构建完返回结果后，清理 `final_parts`、`synthesis_parts`、`section_results` 并调用 `gc.collect()`
- main.py syntax check: Passed

## 2026-05-20 - Comprehensive Memory Optimization: 解决项目运行时内存溢出问题

### Root Cause Analysis
项目运行时内存溢出由以下多个层面的内存管理问题叠加导致：

1. **`ingest_all.py`**: `MAX_CHROMA_BATCH=40000`，允许4万个chunk（含文本+metadata+1024维embedding向量）堆积在内存中才flush，单批次可达200MB+
2. **`vector_store.py`**: `add_documents()` 一次性将全部数据提交给ChromaDB，ChromaDB需在内存中构建HNSW索引
3. **`doc_processor.py`**: `extract_text_from_docx()` 使用O(n²)的段落匹配循环，大文档极慢且CPU/内存高
4. **`build_knowledge_base.py`**: 每文件的所有embedding在内存中累加后才写入向量库，无逐文件清理
5. **`rebuild_medical_standards.py`**: 同上模式
6. **`main.py`**: `await file.read()` 将整个上传文件加载到内存（大文件可达几十MB），且会话历史无单条消息大小限制
7. **`check_all.py`**: `get_all_documents(limit=30000)` 一次性加载3万条文档到内存
8. **`check_medical.py`**: `get_all_documents(limit=20000)` 一次性加载2万条
9. **`check_vector_db.py`**: `get_all_documents(limit=10000)` 一次性加载1万条
10. **`compare_dir.py`**: `get_all_documents(limit=100000)` 一次性加载10万条

### Files Modified

#### 1. `backend/ingest_all.py` - 致命内存堆积修复
- `MAX_CHROMA_BATCH`: 40000 → 1000（单批次内存从~200MB降至~5MB）
- `BATCH_COMMIT_SIZE`: 5 → 1（每个文件处理完立即提交）

#### 2. `backend/vector_store.py` - 添加内部分批提交
- `add_documents()` 现在自动按 batch_size=500 分批提交给 ChromaDB
- 避免 ChromaDB 因单次接收大量数据而 OOM

#### 3. `backend/doc_processor.py` - O(n²)循环修复
- `extract_text_from_docx()` 构建 `elem_to_para` 映射表（O(n)），替换原嵌套循环（O(n²)）
- 大文档处理速度从分钟级降至秒级

#### 4. `backend/build_knowledge_base.py` - 逐文件内存清理
- 添加 `import gc`
- 每个文件处理完后执行 `del text, chunks, embeddings, chunk_ids; gc.collect()`

#### 5. `backend/rebuild_medical_standards.py` - 逐文件内存清理
- 添加 `import gc`
- 每个文件处理完后执行内存清理

#### 6. `backend/build_all_medical.py` - 逐文件内存清理
- 添加 `import gc`，每个文件处理完后清理

#### 7. `backend/quick_build_medical.py` - 逐文件内存清理
- 添加 `import gc`，每个文件处理完后清理

#### 8. `backend/main.py` - 上传流式写入 + 会话历史截断
- 上传改为64KB分块流式写入，避免全量加载到内存
- 上传失败时清理临时文件
- `ConversationHistory` 添加 `max_content_length=10000` 限制单条消息大小

#### 9. `backend/check_all.py` - 分页获取
- 使用分页查询（page_size=2000）替代 `get_all_documents(limit=30000)`

#### 10. `backend/check_medical.py` - 分页获取
- 使用分页查询（page_size=2000）替代 `get_all_documents(limit=20000)`
- 移除重复的分类统计循环

#### 11. `backend/check_vector_db.py` - 分页获取
- 使用分页查询（page_size=2000）替代 `get_all_documents(limit=10000)`

#### 12. `backend/compare_dir.py` - 分页获取
- 使用分页查询（page_size=2000）替代 `get_all_documents(limit=100000)`

---

## 2026-05-20 18:30 - QA Testing: Memory Optimization Verification

### Test Environment
- Conda env: `env_01` at `E:/anaconda/anaconda_content`
- Project: `E:\nrf_sample_codes\working_team_work\project\project_0428`
- ChromaDB: `chroma.sqlite3` = 2.9GB (existing data, NOT rebuilt during tests)

### Test Results Summary

#### Test 1: Syntax Verification (13 files)
All 13 files passed syntax check via `py_compile.compile()`:
main.py, vector_store.py, rag_retriever.py, doc_processor.py, ingest_all.py,
build_knowledge_base.py, rebuild_medical_standards.py, build_all_medical.py,
quick_build_medical.py, check_all.py, check_medical.py, check_vector_db.py, compare_dir.py

#### Test 2: Module Import Verification
- vector_store module: imported successfully
- rag_retriever module: imported successfully
- doc_processor module: imported successfully

#### Test 3: doc_processor.py O(n) Fix Verification
- Real docx file (24877 chars) extracted in 0.21s
- 32 sections, 64 chunks generated
- ALL CHECKS PASSED: extraction, chunking, section parsing all correct

#### Test 4: vector_store.py Batching Logic (6 sub-tests)
- Test 4.1: Empty list → 0 batches (correct)
- Test 4.2: 10 documents → 1 batch (correct)
- Test 4.3: 500 documents → 1 batch (boundary, correct)
- Test 4.4: 501 documents → 2 batches (500 + 1, correct)
- Test 4.5: 1500 documents → 3 batches (500 each, correct)
- Test 4.6: 800 documents with embeddings → 2 batches (500 + 300, correct)
- Result: ALL 6 TESTS PASSED

#### Test 5: ingest_all.py Memory Optimization Verification
- `BATCH_COMMIT_SIZE = 1` (line 244): confirmed — each file flushed immediately
- `MAX_CHROMA_BATCH = 1000` (line 252): confirmed — reduced from 40000
- `flush_pending()` correctly uses `MAX_CHROMA_BATCH` for internal batching (lines 261-262)
- `gc.collect()` called after each flush (line 274)
- Syntax check: PASSED

#### Test 6: FastAPI Backend Integration (8 sub-tests)
- Test 6.1: Config class initialization — PASSED
- Test 6.2: Pydantic models (Message, ChatRequest, ChatResponse, HealthResponse, temperature validation) — PASSED
- Test 6.3: ConversationHistory (add/retrieve, max history cap=5, content truncation 15000→10027, clear, new session) — PASSED
- Test 6.4: doc_processor module import — PASSED
- Test 6.5: vector_store module structure — PASSED
- Test 6.6: rag_retriever module import — PASSED
- Test 6.7: Streaming upload (128KB in 64KB chunks, temp file cleanup) — PASSED
- Test 6.8: FastAPI app structure (9 routes, CORS middleware, ConversationManager) — PASSED
- Result: ALL 8 TESTS PASSED

### Summary
- Total tests: 13 syntax + 3 imports + 1 doc_processor + 6 batching + 1 ingest + 8 integration = 32 checks
- Passed: 32
- Failed: 0

### Note
Full FastAPI server startup with ChromaDB was NOT tested because the existing
`chroma.sqlite3` is 2.9GB and would trigger the original OOM condition. The
optimizations target the ingestion/build pipeline (preventing the database from
growing to problematic sizes), not shrinking an already-oversized database.
The 2.9GB database should be rebuilt using the optimized scripts.

## 2026-05-22 14:45 - Service Started
- Command: `source E:/anaconda/anaconda_content/etc/profile.d/conda.sh && conda activate env_01 && cd "E:/nrf_sample_codes/working_team_work/public/project/project_0428/backend" && python main.py`
- Working Dir: `E:/nrf_sample_codes/working_team_work/public/project/project_0428/backend`
- Purpose: Start the project_0428 FastAPI backend service
- Result: Success - Service running on http://localhost:8000, PID 6048
- Fixed: Copied missing frontend directory from code_writer project

## 2026-05-22 15:40:00 - Analysis
- Topic: project_0428 项目介绍文档生成
- Finding: 项目为医疗器械体系文件审核Agent，基于FastAPI+ChromaDB RAG+MiniMax大模型
- Decision: 全面读取项目README/SPEC/TODOS及核心模块源码后，生成13章节完整介绍文档

## 2026-05-22 15:40:00 - File Edited
- File: E:\nrf_sample_codes\working_team_work\public\docs\code_writer_docs\project_0428_introduction.md
- Change: 创建项目介绍文档（新建），包含项目概述、业务背景、功能特性、技术架构、核心模块、知识库、API、部署、文件结构、工具链、法规标准、开发状态、已知限制等13个章节
- Result: Success - 文件创建完成

## 2026-05-25 09:48:00 - Comprehensive Code Testing Session Started
- Project: E:\nrf_sample_codes\working_team_work\public\project\project_0428
- Task: Full comprehensive code testing - code review + existing test execution + static analysis
- Scope: backend/main.py, backend/rag_retriever.py, backend/doc_processor.py, backend/vector_store.py, frontend/index.html, all test files

## 2026-05-25 09:48:30 - Bash Command Executed
- Command: `ls -la "E:/nrf_sample_codes/working_team_work/public/project/project_0428/"`
- Working Dir: E:\nrf_sample_codes\working_team_work\code_test
- Purpose: List project root directory structure
- Result: Success - 17 items listed

## 2026-05-25 09:48:30 - Bash Command Executed
- Command: `find "E:/nrf_sample_codes/working_team_work/public/project/project_0428/" -type f -name "*.py" -o -name "*.html" ...`
- Working Dir: E:\nrf_sample_codes\working_team_work\code_test
- Purpose: List all source files in the project
- Result: Success - 40 source files found

## 2026-05-25 09:49:00 - Analysis
- Topic: Core source code review (all 4 main modules + frontend)
- Finding: Multiple issues identified including hardcoded API key, XSS risks, random vector fallback
- Decision: Proceed with test execution then compile comprehensive report

## 2026-05-25 09:50:00 - Bash Command Executed
- Command: `python -m pytest tests/test_doc_processor.py -v`
- Working Dir: E:/nrf_sample_codes/working_team_work/public/project/project_0428/backend
- Purpose: Run existing doc_processor unit tests
- Result: Success - 10/10 passed

## 2026-05-25 09:50:00 - Bash Command Executed
- Command: `python test_main_integration.py`
- Working Dir: E:/nrf_sample_codes/working_team_work/public/project/project_0428/backend
- Purpose: Run integration tests (config, models, history, module imports, app structure)
- Result: Success - 8/8 passed

## 2026-05-25 09:50:00 - Bash Command Executed
- Command: `python test_vector_store_batching.py`
- Working Dir: E:/nrf_sample_codes/working_team_work/public/project/project_0428/backend
- Purpose: Run vector store batching logic tests
- Result: Success - 6/6 passed

## 2026-05-25 09:55:00 - Analysis
- Topic: Comprehensive code review findings
- Finding: Identified 3 critical, 4 high, 6 medium, 5 low severity issues. Major categories: security (API key leak, XSS, no auth), robustness (fail-open embedding, COM automation), code quality (magic numbers, code duplication)
- Decision: Report compiled and saved to E:\nrf_sample_codes\working_team_work\public\docs\code_test_docs\code-test-report-project_0428-2026-05-25.md

## 2026-05-29 15:24:42 - File Created
- File: docs/项目介绍.md
- Change: 创建项目介绍文档，包含项目概述、整体结构、技术路线、代码文件详解（18 个 Python 文件）、API 接口清单、知识库数据、开发历史、部署指南等完整内容
- Result: Success - 文件已创建于 E:\nrf_sample_codes\working_team_work\public\project\project_0428_beta\docs\项目介绍.md
