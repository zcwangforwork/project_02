"""
医疗器械体系文件审核 - 文档处理模块
支持 Word (.docx) 和 PDF 文件的文本提取与分块
输出结构化 Markdown，保留标题层级、表格、列表
"""
import os
import re
from typing import List, Tuple
from pathlib import Path


# ============== Word 标题样式到 Markdown 层级映射 ==============
HEADING_STYLE_MAP = {
    'heading 1': 1, 'heading 2': 2, 'heading 3': 3,
    'heading 4': 4, 'heading 5': 5, 'heading 6': 6,
    '标题 1': 1, '标题 2': 2, '标题 3': 3,
    '标题 4': 4, '标题 5': 5, '标题 6': 6,
    'title': 1,
}


def extract_text_from_docx(file_path: str) -> str:
    """
    从 Word 文档提取结构化 Markdown 文本

    保留标题层级（#/##/###）、表格、列表，供 LLM 理解文档结构

    Args:
        file_path: .docx 文件路径

    Returns:
        结构化 Markdown 文本
    """
    try:
        from docx import Document
        doc = Document(file_path)
        parts = []

        # 构建 element -> paragraph 的快速查找映射（O(1) 替代原 O(n²) 循环）
        elem_to_para = {}
        for p in doc.paragraphs:
            try:
                elem_to_para[id(p._element)] = p
            except Exception:
                pass

        for element in doc.element.body:
            tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag

            if tag == 'p':
                # 段落处理 - O(1) 查找
                para = elem_to_para.get(id(element))
                if para is None:
                    continue

                text = para.text.strip()
                if not text:
                    parts.append('')
                    continue

                # 检测标题样式
                style_name = (para.style.name or '').lower() if para.style else ''
                heading_level = HEADING_STYLE_MAP.get(style_name, 0)

                if heading_level > 0:
                    parts.append(f'{"#" * heading_level} {text}')
                elif _is_list_paragraph(para):
                    parts.append(_format_list_item(para, text))
                else:
                    parts.append(text)

            elif tag == 'tbl':
                # 表格处理
                table_md = _extract_table_from_element(element, doc)
                if table_md:
                    parts.append('')
                    parts.append(table_md)
                    parts.append('')

        return '\n'.join(parts)
    except ImportError:
        raise ImportError("请安装 python-docx: pip install python-docx")


def _is_list_paragraph(para) -> bool:
    """判断段落是否为列表项"""
    style_name = (para.style.name or '').lower() if para.style else ''
    list_keywords = ['list', 'listparagraph', '列表']
    return any(kw in style_name for kw in list_keywords)


def _format_list_item(para, text: str) -> str:
    """格式化列表项为 Markdown"""
    style_name = (para.style.name or '').lower() if para.style else ''
    # 判断有序/无序
    if 'number' in style_name or re.match(r'^\d+[.、）)]', text):
        # 有序列表：提取数字前缀或自动编号
        match = re.match(r'^(\d+)[.、）)]\s*', text)
        if match:
            return f"{match.group(1)}. {text[match.end():]}"
        return f"1. {text}"
    else:
        return f"- {text}"


def _extract_table_from_element(table_element, doc) -> str:
    """从 XML 元素提取表格并转为 Markdown 格式"""
    rows = []
    for row_elem in table_element.iterchildren():
        tag = row_elem.tag.split('}')[-1] if '}' in row_elem.tag else row_elem.tag
        if tag != 'tr':
            continue
        cells = []
        for cell_elem in row_elem.iterchildren():
            cell_tag = cell_elem.tag.split('}')[-1] if '}' in cell_elem.tag else cell_elem.tag
            if cell_tag != 'tc':
                continue
            # 提取单元格文本
            cell_text_parts = []
            for p_elem in cell_elem.iterchildren():
                p_tag = p_elem.tag.split('}')[-1] if '}' in p_elem.tag else p_elem.tag
                if p_tag == 'p':
                    texts = []
                    for t_elem in p_elem.iter():
                        t_tag = t_elem.tag.split('}')[-1] if '}' in t_elem.tag else t_elem.tag
                        if t_tag == 't':
                            texts.append(t_elem.text or '')
                    cell_text_parts.append(''.join(texts).strip())
            cells.append(' | '.join(cell_text_parts) if cell_text_parts else ' ')

        if cells:
            rows.append(cells)

    if not rows:
        return ''

    # 统一列数
    max_cols = max(len(r) for r in rows) if rows else 0
    for r in rows:
        while len(r) < max_cols:
            r.append(' ')

    # 构建 Markdown 表格
    md_lines = []
    # 表头
    md_lines.append('| ' + ' | '.join(rows[0]) + ' |')
    # 分隔行
    md_lines.append('| ' + ' | '.join(['---'] * max_cols) + ' |')
    # 数据行
    for row in rows[1:]:
        md_lines.append('| ' + ' | '.join(row) + ' |')

    return '\n'.join(md_lines)


def extract_text_from_pdf(file_path: str) -> str:
    """
    从 PDF 文件提取文本内容，尝试用字体大小识别标题层级

    Args:
        file_path: .pdf 文件路径

    Returns:
        结构化 Markdown 文本
    """
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue

                # 尝试提取字体信息来识别标题
                chars = page.chars
                if chars:
                    text = _pdf_text_with_headings(text, chars)

                text_parts.append(text)
        return '\n'.join(text_parts)
    except ImportError:
        raise ImportError("请安装 pdfplumber: pip install pdfplumber")


def _pdf_text_with_headings(text: str, chars: list) -> str:
    """
    根据 PDF 字符的字体大小，启发式识别标题并转为 Markdown

    逻辑：统计所有字体大小，最大的视为标题层级
    """
    if not chars:
        return text

    # 统计字体大小分布
    size_count = {}
    for c in chars:
        size = round(c.get('size', 0), 1)
        if size > 0:
            size_count[size] = size_count.get(size, 0) + len(c.get('text', ''))

    if not size_count:
        return text

    # 按大小降序排列，前3个尺寸视为标题
    sorted_sizes = sorted(size_count.keys(), reverse=True)
    body_size = sorted_sizes[-1] if sorted_sizes else 10.0  # 最小的通常是正文

    heading_map = {}
    heading_level = 1
    for size in sorted_sizes:
        if size > body_size * 1.15 and heading_level <= 3:  # 比正文大15%以上视为标题
            heading_map[size] = heading_level
            heading_level += 1
        else:
            break

    if not heading_map:
        return text

    # 按行处理，识别标题行
    lines = text.split('\n')
    result = []
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            result.append('')
            continue

        # 检查该行是否主要是大字体
        line_chars = [c for c in chars if c.get('text', '').strip()
                      and c['text'].strip() in line_stripped]
        if line_chars:
            # 取该行最多的字体大小
            line_sizes = {}
            for c in line_chars:
                s = round(c.get('size', 0), 1)
                line_sizes[s] = line_sizes.get(s, 0) + 1
            dominant_size = max(line_sizes, key=line_sizes.get) if line_sizes else 0

            if dominant_size in heading_map:
                level = heading_map[dominant_size]
                result.append(f'{"#" * level} {line_stripped}')
                continue

        result.append(line_stripped)

    return '\n'.join(result)


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
    根据文件扩展名自动识别并提取文本（结构化 Markdown 格式）

    Args:
        file_path: 文件路径

    Returns:
        结构化 Markdown 文本
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


def split_by_markdown_headers(text: str) -> List[Tuple[str, str, int]]:
    """
    按 Markdown 标题层级分割文档

    Args:
        text: Markdown 格式的文档文本

    Returns:
        [(标题, 内容, 层级), ...] 段落列表
        层级: 1=一级标题, 2=二级标题, 3=三级标题...
    """
    sections = []
    current_title = "文档开头"
    current_level = 0
    current_content = []

    for line in text.split('\n'):
        # 检测 Markdown 标题行
        match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if match:
            # 保存前一个段落
            if current_content:
                full_content = '\n'.join(current_content).strip()
                if full_content:
                    sections.append((current_title, full_content, current_level))

            current_level = len(match.group(1))
            current_title = match.group(2).strip()
            current_content = []
        else:
            current_content.append(line)

    # 保存最后一个段落
    if current_content:
        full_content = '\n'.join(current_content).strip()
        if full_content:
            sections.append((current_title, full_content, current_level))

    # 如果没有检测到任何标题，把整篇文档作为一个段落
    if not sections and text.strip():
        sections.append(("完整文档", text.strip(), 0))

    return sections


# ============== 编号式标题正则（多策略大纲解析使用）==============
# 顺序: 优先匹配更具体的格式，避免误判
# 层级设计原则：
#   - "第X章" 和 阿拉伯数字一级编号 → L1（最高级）
#   - "X.X" → L2（与 Markdown ## 同级，作为审核单元）
#   - "X.X.X" / 中文 "一、" → L3（合并到父二级小节内）
#   - "X.X.X.X" / "(一)" / "(1)" → L4（更深层，合并）
# 这样在与 Markdown # ## ### 混合出现时，编号标题通常会成为子节而非平级
_NUMBERING_PATTERNS = [
    # "第X章" / "第X篇" / "第X部分" → level 1
    (re.compile(r'^(第[一二三四五六七八九十百千零〇\d]+[章篇部分])[\s::、]*(.*)$'), 1),
    # "X.X.X.X" 四级编号 → level 4
    (re.compile(r'^(\d+\.\d+\.\d+\.\d+)[\s::、)]*(.+)$'), 4),
    # "X.X.X" 三级编号 → level 3
    (re.compile(r'^(\d+\.\d+\.\d+)[\s::、)]*(.+)$'), 3),
    # "X.X" 二级编号 → level 2
    (re.compile(r'^(\d+\.\d+)[\s::、)]*(.+)$'), 2),
    # "X、" 或 "X." 一级数字编号 → level 1（前提：后面有中文/字母标题文字，且长度合理）
    (re.compile(r'^(\d+)[、.)][\s]+([^\d].{1,80})$'), 1),
    # "一、二、三、" 中文数字 → level 3（通常出现在 ## 小节内，作为列举项；不抢占 L2 审核单元位置）
    (re.compile(r'^([一二三四五六七八九十]+)[、.)][\s]*(.{1,80})$'), 3),
    # "(一)(二)" 或 "（一）（二）" 中文带括号 → level 4
    (re.compile(r'^[\(（]([一二三四五六七八九十]+)[\)）][\s]*(.{1,80})$'), 4),
    # "(1)(2)" 或 "（1）（2）" 数字带括号 → level 4
    (re.compile(r'^[\(（](\d+)[\)）][\s]*(.{1,80})$'), 4),
]


def _detect_numbering_heading(line: str) -> Tuple[int, str]:
    """
    检测一行是否为编号式标题。

    Returns:
        (level, title) — level=0 表示非标题；title 为标题文字（含编号前缀）
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        # 标题一般不会太长（>120字符基本是正文）
        return 0, ""
    # 已经是 Markdown 标题的不再二次识别
    if stripped.startswith('#'):
        return 0, ""

    for pattern, level in _NUMBERING_PATTERNS:
        if pattern.match(stripped):
            return level, stripped
    return 0, ""


def parse_document_outline(text: str) -> List[dict]:
    """
    多策略大纲解析：识别文档的章节树状结构。

    策略融合（按优先级）：
    1. Markdown 标题（# / ## / ### ...）— 来自 Word Heading 样式或显式 Markdown
    2. 编号正则（"第X章" / "X.X" / "X.X.X" / "一、" / "(一)" / "(1)" 等）

    Returns:
        树状大纲列表，每个节点：
        {
            "title": str,           # 标题文字
            "level": int,           # 层级 1~6
            "content": str,         # 该标题下的直接正文（不含子节点的正文）
            "children": List[dict]  # 子节点
        }
    """
    # 第一步：扫描每一行，识别"标题行"和"正文行"
    # 标题行: {"is_heading": True, "level": int, "title": str}
    # 正文行: {"is_heading": False, "text": str}
    items = []
    for raw_line in text.split('\n'):
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            items.append({"is_heading": False, "text": ""})
            continue

        # 策略1: Markdown 标题
        md_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
        if md_match:
            items.append({
                "is_heading": True,
                "level": len(md_match.group(1)),
                "title": md_match.group(2).strip()
            })
            continue

        # 策略2: 编号式标题
        num_level, num_title = _detect_numbering_heading(line)
        if num_level > 0:
            items.append({
                "is_heading": True,
                "level": num_level,
                "title": num_title
            })
            continue

        items.append({"is_heading": False, "text": line})

    # 第二步：把扁平的 items 按层级组装成树
    root_children: List[dict] = []
    # 用栈维护"当前层级路径"，stack[-1] 是最近的祖先节点
    stack: List[dict] = []
    # 一个虚拟根，方便统一处理
    pending_preface: List[str] = []  # 第一个标题之前的正文

    def _new_node(level: int, title: str) -> dict:
        return {"title": title, "level": level, "content": "", "children": []}

    for item in items:
        if item["is_heading"]:
            node = _new_node(item["level"], item["title"])
            # 找到合适的父：栈中第一个 level < 当前 level 的
            while stack and stack[-1]["level"] >= node["level"]:
                stack.pop()
            if stack:
                stack[-1]["children"].append(node)
            else:
                root_children.append(node)
            stack.append(node)
        else:
            # 正文行 → 加到当前栈顶节点的 content；若栈空则放 pending_preface
            text_line = item.get("text", "")
            if stack:
                if stack[-1]["content"]:
                    stack[-1]["content"] += "\n" + text_line
                else:
                    stack[-1]["content"] = text_line
            else:
                pending_preface.append(text_line)

    # 第三步：如果存在 preface（文档开头未归属任何标题的正文），合成一个虚拟节点
    preface_text = "\n".join(pending_preface).strip()
    if preface_text:
        root_children.insert(0, {
            "title": "文档开头",
            "level": 1,
            "content": preface_text,
            "children": []
        })

    # 第四步：清理每个节点 content 的首尾空白
    def _clean(node: dict):
        node["content"] = node["content"].strip()
        for child in node["children"]:
            _clean(child)
    for n in root_children:
        _clean(n)

    return root_children


def _render_subtree_as_content(node: dict, base_level: int = 2) -> str:
    """
    把一个节点及其所有子节点（三级及更深）的内容渲染为合并文本。
    用于"二级小节"作为审核单元时，把它下面的三级/四级内容并入。

    Args:
        node: 起始节点
        base_level: 起始层级（用于决定子节点用几个 # 做副标题）

    Returns:
        合并后的文本（保留子层级的小标题，便于 LLM 理解结构）
    """
    parts: List[str] = []
    if node.get("content"):
        parts.append(node["content"])

    for child in node.get("children", []):
        sub_level = child.get("level", base_level + 1)
        # 渲染为副标题前缀（用 # 标记，便于 LLM 识别）
        hashes = "#" * min(sub_level, 6)
        parts.append("")
        parts.append(f"{hashes} {child['title']}")
        sub_text = _render_subtree_as_content(child, sub_level)
        if sub_text:
            parts.append(sub_text)

    return "\n".join(p for p in parts if p is not None).strip()


def flatten_to_audit_units(outline: List[dict]) -> List[Tuple[str, str, int, str]]:
    """
    把树状大纲扁平化为审核单元列表。

    规则（与本次需求一致）:
    - 审核粒度为"二级小节"
    - 三级及以下作为内容合并入父二级小节
    - 一级章节本身：若其下有二级子节 → 不作为独立审核单元（其直接 content 并入第一个子节前），
      若其下没有二级子节 → 一级章节本身作为审核单元（合并其下三级及以下内容）

    Args:
        outline: parse_document_outline() 的返回值

    Returns:
        [(title, content, level, breadcrumb), ...]
        breadcrumb: "第一章 / 1.1 标题" 形式的面包屑，便于 LLM 理解上下文
    """
    units: List[Tuple[str, str, int, str]] = []

    def _walk_level1(chap_node: dict):
        chap_title = chap_node["title"]
        # 找到所有 level==2 的子节
        level2_children = [c for c in chap_node["children"] if c.get("level") == 2]
        # 也允许"level 不是 2 但 < 当前 chap+2"的情况下走二级处理（兼容跳级）
        if not level2_children:
            # 没有二级 → 整章作为审核单元，合并所有子节内容
            content_parts = []
            if chap_node.get("content"):
                content_parts.append(chap_node["content"])
            for child in chap_node["children"]:
                hashes = "#" * min(child.get("level", 2), 6)
                content_parts.append(f"\n{hashes} {child['title']}")
                sub = _render_subtree_as_content(child, child.get("level", 2))
                if sub:
                    content_parts.append(sub)
            merged = "\n".join(p for p in content_parts if p).strip()
            if merged or chap_node.get("children"):
                units.append((chap_title, merged, 1, chap_title))
            return

        # 有二级子节 → 一级章节本身的 content 作为"章引言"，并入第一个二级小节前
        chap_preface = chap_node.get("content", "").strip()

        for idx, sec_node in enumerate(level2_children):
            sec_title = sec_node["title"]
            sec_parts = []
            # 把章引言并入第一个小节（仅一次）
            if idx == 0 and chap_preface:
                sec_parts.append(f"[本章引言]\n{chap_preface}\n")

            if sec_node.get("content"):
                sec_parts.append(sec_node["content"])

            # 把该二级小节下的所有子节（三级及更深）合并进来
            for sub in sec_node.get("children", []):
                hashes = "#" * min(sub.get("level", 3), 6)
                sec_parts.append(f"\n{hashes} {sub['title']}")
                sub_text = _render_subtree_as_content(sub, sub.get("level", 3))
                if sub_text:
                    sec_parts.append(sub_text)

            merged = "\n".join(p for p in sec_parts if p).strip()
            breadcrumb = f"{chap_title} / {sec_title}"
            units.append((sec_title, merged, 2, breadcrumb))

    for top_node in outline:
        top_level = top_node.get("level", 1)
        if top_level == 1:
            _walk_level1(top_node)
        else:
            # 顶层不是 level 1（罕见，比如文档只有 ## 没有 #）
            # 当 level==2 时，每个都直接作为审核单元
            if top_level == 2:
                content_parts = []
                if top_node.get("content"):
                    content_parts.append(top_node["content"])
                for sub in top_node.get("children", []):
                    hashes = "#" * min(sub.get("level", 3), 6)
                    content_parts.append(f"\n{hashes} {sub['title']}")
                    sub_text = _render_subtree_as_content(sub, sub.get("level", 3))
                    if sub_text:
                        content_parts.append(sub_text)
                merged = "\n".join(p for p in content_parts if p).strip()
                units.append((top_node["title"], merged, 2, top_node["title"]))
            else:
                # level >= 3 直接作为独立审核单元（极少出现）
                merged = _render_subtree_as_content(top_node, top_level)
                if top_node.get("content"):
                    merged = (top_node["content"] + "\n" + merged).strip()
                units.append((top_node["title"], merged, top_level, top_node["title"]))

    return units


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
