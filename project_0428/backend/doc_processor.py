"""
医疗器械体系文件审核 - 文档处理模块
支持 Word (.docx) 和 PDF 文件的文本提取与分块
"""
import os
from typing import List, Tuple
from pathlib import Path


def extract_text_from_docx(file_path: str) -> str:
    """
    从 Word 文档提取文本内容

    Args:
        file_path: .docx 文件路径

    Returns:
        提取的文本内容
    """
    try:
        from docx import Document
        doc = Document(file_path)
        paragraphs = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)
        return '\n'.join(paragraphs)
    except ImportError:
        raise ImportError("请安装 python-docx: pip install python-docx")


def extract_text_from_pdf(file_path: str) -> str:
    """
    从 PDF 文件提取文本内容

    Args:
        file_path: .pdf 文件路径

    Returns:
        提取的文本内容
    """
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return '\n'.join(text_parts)
    except ImportError:
        raise ImportError("请安装 pdfplumber: pip install pdfplumber")


def extract_text_from_doc(file_path: str) -> str:
    """
    从旧版 Word .doc 文件提取文本内容
    优先使用 win32com (COM自动化)，回退到 olefile
    """
    # 方法1: win32com (需要 Windows + MS Word)
    try:
        import win32com.client
        import pythoncom
        pythoncom.CoInitialize()
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        try:
            abs_path = os.path.abspath(file_path)
            doc = word.Documents.Open(abs_path, ReadOnly=True)
            text = doc.Content.Text
            doc.Close(False)
            word.Quit()
            pythoncom.CoUninitialize()
            if text and text.strip():
                return text.strip()
        except Exception:
            try:
                word.Quit()
            except Exception:
                pass
            pythoncom.CoUninitialize()
    except Exception:
        pass

    # 方法2: olefile 提取
    try:
        import olefile
        ole = olefile.OleFileIO(file_path)
        for stream_name in ['1Table', '0Table', 'WordDocument']:
            if ole.exists(stream_name):
                data = ole.openstream(stream_name).read()
                try:
                    text = data.decode('utf-16le', errors='ignore')
                    clean = ''.join(c for c in text if c.isprintable() or c in '\n\r\t')
                    if len(clean) > 50:
                        ole.close()
                        return clean
                except Exception:
                    pass
        ole.close()
    except Exception:
        pass

    # 方法3: 二进制扫描
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        text_parts = []
        i = 0
        while i < len(data) - 1:
            char_code = data[i] | (data[i+1] << 8)
            if 0x4e00 <= char_code <= 0x9fff or 0x3000 <= char_code <= 0x303f or char_code in (0x000d, 0x000a):
                text_parts.append(chr(char_code))
                i += 2
            elif 0x0020 <= char_code <= 0x007e:
                text_parts.append(chr(char_code))
                i += 2
            else:
                if text_parts and text_parts[-1] != '\n':
                    text_parts.append('\n')
                i += 1
        result = ''.join(text_parts).strip()
        if len(result) > 50:
            return result
    except Exception:
        pass

    return ""


def extract_text_from_txt(file_path: str) -> str:
    """从文本文件提取内容，自动检测编码"""
    for enc in ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    return ""


def extract_text(file_path: str) -> str:
    """
    根据文件扩展名自动识别并提取文本

    Args:
        file_path: 文件路径

    Returns:
        提取的文本内容
    """
    ext = Path(file_path).suffix.lower()
    if ext == '.docx':
        return extract_text_from_docx(file_path)
    elif ext == '.doc':
        return extract_text_from_doc(file_path)
    elif ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.txt':
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}，仅支持 .docx, .doc, .pdf 和 .txt")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    将长文本分块，便于向量检索

    Args:
        text: 待分块的文本
        chunk_size: 每块字符数
        overlap: 相邻块重叠字符数

    Returns:
        分块后的文本列表
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunks.append(text[start:])
            break

        # 在句号、换行或逗号处截断，保证语义完整
        chunk = text[start:end]
        for sep in ['\n\n', '\n', '。', '；', '，', '. ', '; ', ', ']:
            last_sep = chunk.rfind(sep)
            if last_sep > chunk_size * 0.5:
                end = start + last_sep + len(sep)
                chunk = chunk[:last_sep + len(sep)]
                break

        chunks.append(chunk)
        start = end - overlap

    return chunks


def get_file_metadata(file_path: str) -> dict:
    """
    获取文件元数据

    Args:
        file_path: 文件路径

    Returns:
        包含文件信息的字典
    """
    path = Path(file_path)
    return {
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
    }
