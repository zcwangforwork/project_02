"""
医疗器械体系文件审核 Agent - FastAPI 后端服务
"""
import os
import json
import base64
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from pathlib import Path
import tempfile

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# ============== 配置管理 ==============
API_KEY = "ark-6c047509-3ac1-4689-b796-47363425012c-3112a"


class Config:
    """配置管理类"""
    def __init__(self):
        self.api_url = os.getenv("OPENAI_API_URL", "https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions")
        self.embedding_url = os.getenv("EMBEDDING_API_URL", "https://ark.cn-beijing.volces.com/api/coding/v3/embeddings/multimodal")
        self.api_key = API_KEY or os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("OPENAI_MODEL", "glm-5.1")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "doubao-embedding-vision-250615")
        self.timeout = float(os.getenv("REQUEST_TIMEOUT", "60"))

    def to_dict(self):
        return {
            "api_url": self.api_url,
            "model": self.model,
            "timeout": self.timeout,
            "configured": bool(self.api_key)
        }


config = Config()


# ============== Pydantic 模型 ==============
class Message(BaseModel):
    """聊天消息模型"""
    role: str = Field(default="user", description="角色: user/assistant/system")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    """聊天请求模型"""
    messages: List[Message] = Field(..., description="消息列表")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: int = Field(default=32000, ge=1, le=32000, description="最大令牌数")


class ChatResponse(BaseModel):
    """聊天响应模型"""
    answer: str = Field(..., description="助手的回答")
    usage: Optional[Dict[str, int]] = Field(None, description="令牌使用情况")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    config: Dict
    vectorstore_loaded: bool = False
    document_count: int = 0


# ============== 全局变量 ==============
vector_store = None
rag_retriever = None


# ============== 医疗器械 System Prompt ==============
MEDICAL_DEVICE_SYSTEM_PROMPT = """你是一个专业的医疗器械企业体系文件审核专家。

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


# ============== 会话管理 ==============
class ConversationHistory:
    """简单的会话历史管理"""
    def __init__(self, max_history: int = 10):
        self.history: Dict[str, List[Dict]] = {}
        self.max_history = max_history

    def get_or_create(self, session_id: str = "default") -> List[Dict]:
        if session_id not in self.history:
            self.history[session_id] = []
        return self.history[session_id]

    def add_message(self, session_id: str, role: str, content: str):
        messages = self.get_or_create(session_id)
        messages.append({"role": role, "content": content})

        if len(messages) > self.max_history:
            system_msg = [m for m in messages if m["role"] == "system"]
            other_msgs = [m for m in messages if m["role"] != "system"]
            self.history[session_id] = system_msg + other_msgs[-self.max_history:]

    def clear(self, session_id: str = "default"):
        if session_id in self.history:
            del self.history[session_id]


conversation_manager = ConversationHistory()


# ============== 初始化向量存储 ==============
def init_vector_store():
    """初始化向量存储和 RAG 检索器"""
    global vector_store, rag_retriever

    try:
        from vector_store import create_vector_store, MiniMaxEmbeddingFunction
        from rag_retriever import create_rag_retriever

        # 使用绝对路径确保在任何工作目录下都能正确加载
        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(base_dir, "data", "chroma_db")

        # 创建 embedding function（查询时用于生成 query embedding）
        embedding_function = MiniMaxEmbeddingFunction(
            api_key=config.api_key,
            api_url=config.embedding_url,
            model=config.embedding_model,
            dimension=1024
        )

        # 不传 embedding_function 初始化，避免启动时加载大 HNSW 索引
        # 查询时通过 query_texts 会自动调用 embedding_function
        vector_store = create_vector_store(persist_directory=db_path, embedding_function=embedding_function)
        rag_retriever = create_rag_retriever(
            vector_store=vector_store,
            api_key=config.api_key,
            api_url=config.api_url,
            model=config.model
        )
        return True
    except Exception as e:
        print(f"向量存储初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============== API 调用函数 ==============
async def call_openai_api(messages: List[Dict], temperature: float = 0.7, max_tokens: int = 32000) -> Dict:
    """调用 OpenAI 兼容 API"""
    if not config.api_key:
        raise HTTPException(status_code=500, detail="API Key 未配置")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}"
    }

    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    async with httpx.AsyncClient(timeout=config.timeout, trust_env=False) as client:
        try:
            response = await client.post(config.api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="请求超时")
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json() if e.response.content else {"error": str(e)}
            raise HTTPException(status_code=e.response.status_code, detail=error_detail)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"API 调用失败: {str(e)}")


async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """获取文本嵌入向量"""
    if not config.api_key:
        raise HTTPException(status_code=500, detail="API Key 未配置")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}"
    }

    embeddings = []
    async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
        for text in texts:
            payload = {
                "model": config.embedding_model,
                "encoding_format": "float",
                "input": [{"text": text[:8000], "type": "text"}]
            }
            try:
                response = await client.post(config.embedding_url, headers=headers, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    data = result.get("data", {})
                    if data:
                        embeddings.append(data.get("embedding", []))
                    else:
                        embeddings.append([])
                else:
                    embeddings.append([])
            except Exception:
                embeddings.append([])

    return embeddings


# ============== FastAPI 应用 ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("医疗器械体系文件审核 Agent 后端服务启动")
    print(f"API URL: {config.api_url}")
    print(f"Model: {config.model}")

    # 初始化向量存储
    vs_loaded = init_vector_store()
    if vs_loaded and vector_store:
        doc_count = vector_store.count()
        print(f"向量库已加载: {doc_count} 个文档")
    else:
        print("警告: 向量库未加载，部分功能可能不可用")

    yield

    print("医疗器械体系文件审核 Agent 后端服务关闭")


app = FastAPI(
    title="医疗器械体系文件审核 Agent API",
    description="医疗器械企业体系文件智能审核助手",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")


# ============== 前端路由 ==============
@app.get("/")
async def serve_frontend():
    """服务前端页面"""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ============== API 路由 ==============
@app.get("/info", response_model=Dict)
async def root():
    """根路径 - 服务信息"""
    return {
        "name": "医疗器械体系文件审核 Agent API",
        "version": "2.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    doc_count = vector_store.count() if vector_store else 0
    return HealthResponse(
        status="healthy",
        config=config.to_dict(),
        vectorstore_loaded=vector_store is not None,
        document_count=doc_count
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, session_id: str = "default"):
    """聊天接口 - 通用问答模式"""
    all_messages = [{"role": "system", "content": MEDICAL_DEVICE_SYSTEM_PROMPT}]

    history = conversation_manager.get_or_create(session_id)
    all_messages.extend(history)

    for msg in request.messages:
        all_messages.append({"role": msg.role, "content": msg.content})

    result = await call_openai_api(
        messages=all_messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )

    # OpenAI 兼容格式响应解析
    answer = ""
    choices = result.get("choices", [])
    if choices and len(choices) > 0:
        choice = choices[0]
        message = choice.get("message", {})
        answer = message.get("content", "")

    if not answer:
        answer = str(result) if result else ""

    for msg in request.messages:
        conversation_manager.add_message(session_id, msg.role, msg.content)
    conversation_manager.add_message(session_id, "assistant", answer)

    return ChatResponse(answer=answer, usage=result.get("usage"))


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form("default")
):
    """
    上传体系文件并提取文本内容

    Args:
        file: 上传的文件（.docx 或 .pdf）
        session_id: 会话 ID

    Returns:
        提取的文档内容和基本信息
    """
    # 检查文件类型
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in ['.docx', '.pdf']:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}，仅支持 .docx 和 .pdf"
        )

    # 保存上传文件到临时目录
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 提取文本
        from doc_processor import extract_text, get_file_metadata

        text = extract_text(tmp_path)
        metadata = get_file_metadata(tmp_path)
        metadata["filename"] = filename

        if not text or len(text.strip()) < 50:
            raise HTTPException(status_code=400, detail="文档内容过少或无法提取文本")

        # 保存到会话历史
        conversation_manager.add_message(session_id, "user", f"[上传文件: {filename}]\n{text[:5000]}")
        conversation_manager.add_message(session_id, "assistant", f"已收到文件: {filename}，文档长度: {len(text)} 字符。请问您想如何处理这个文件？")

        return {
            "filename": filename,
            "text_length": len(text),
            "text_preview": text[:1000],
            "metadata": metadata,
            "session_id": session_id
        }

    finally:
        # 删除临时文件
        os.unlink(tmp_path)


@app.post("/api/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    question: str = Form("请审核这份体系文件，给出修改建议"),
    session_id: str = Form("default"),
    audit_type: str = Form("risk_management")
):
    """
    使用 RAG 技术分析体系文件

    Args:
        file: 上传的文件（.docx 或 .pdf）
        question: 用户的问题或指令
        session_id: 会话 ID
        audit_type: 审核类型，"risk_management"（风险管理专项对照审核）或 "general"（综合体系审核）

    Returns:
        审核结果
    """
    if not rag_retriever:
        raise HTTPException(status_code=503, detail="向量库未加载，请稍后重试")

    # 检查文件类型
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in ['.docx', '.pdf']:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}")

    # 验证审核类型
    if audit_type not in ["risk_management", "general"]:
        audit_type = "risk_management"

    # 保存上传文件到临时目录
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from doc_processor import extract_text

        # 提取文档文本
        text = extract_text(tmp_path)
        if not text:
            raise HTTPException(status_code=400, detail="无法提取文档文本")

        # 使用 RAG 分析文档（使用分段对照模式）
        use_contrast_mode = (audit_type == "risk_management")
        result = await rag_retriever.analyze_document(
            user_document=text,
            user_filename=filename,
            use_contrast_mode=use_contrast_mode,
            audit_type=audit_type
        )

        # 构建检索到的文档信息
        retrieved_docs_info = []
        for doc in result.get("retrieved_docs", []):
            retrieved_docs_info.append({
                "source": doc.get("source", "未知"),
                "preview": doc.get("text", "")[:200]
            })

        # 保存到会话历史
        conversation_manager.add_message(session_id, "user", f"[上传文件分析: {filename}]")
        conversation_manager.add_message(session_id, "assistant", result["answer"])

        return {
            "filename": filename,
            "answer": result["answer"],
            "usage": result.get("usage"),
            "retrieved_docs": retrieved_docs_info,
            "audit_type": audit_type,
            "contrast_mode": use_contrast_mode
        }

    finally:
        os.unlink(tmp_path)


@app.post("/api/clear")
async def clear_history(session_id: str = "default"):
    """清除会话历史"""
    conversation_manager.clear(session_id)
    return {"message": "会话历史已清除", "session_id": session_id}


@app.get("/api/history/{session_id}")
async def get_history(session_id: str = "default"):
    """获取会话历史"""
    history = conversation_manager.get_or_create(session_id)
    return {"session_id": session_id, "history": history}


@app.get("/api/vectorstore/status")
async def vectorstore_status():
    """获取向量库状态"""
    if not vector_store:
        return {"loaded": False, "document_count": 0}

    return {
        "loaded": True,
        "document_count": vector_store.count()
    }


# ============== 运行入口 ==============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
