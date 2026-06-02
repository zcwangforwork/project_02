"""
测试 doc_processor.py 的文档结构提取功能
"""
import sys
import os

# 添加 backend 目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from doc_processor import split_by_markdown_headers


class TestSplitByMarkdownHeaders:
    """测试 Markdown 标题分割"""

    def test_basic_headings(self):
        """基本的 #/##/### 标题分割"""
        text = """# 第一章 风险管理计划
这是第一章的内容。

## 1.1 适用范围
这是1.1的内容。

## 1.2 职责分配
这是1.2的内容。

# 第二章 危害识别
这是第二章的内容。"""

        sections = split_by_markdown_headers(text)
        assert len(sections) == 4
        assert sections[0][0] == "第一章 风险管理计划"
        assert sections[0][2] == 1  # level
        assert sections[1][0] == "1.1 适用范围"
        assert sections[1][2] == 2
        assert sections[2][0] == "1.2 职责分配"
        assert sections[3][0] == "第二章 危害识别"
        assert sections[3][2] == 1

    def test_no_headings(self):
        """没有标题的文档，整体作为一个段落"""
        text = """这是第一段内容。
这是第二段内容。
没有标题的纯文本。"""

        sections = split_by_markdown_headers(text)
        assert len(sections) == 1
        assert sections[0][2] == 0  # level 0 = no heading

    def test_deeply_nested_headers(self):
        """深层嵌套标题"""
        text = """# 一级标题
一级内容
## 二级标题
二级内容
### 三级标题
三级内容
#### 四级标题
四级内容"""

        sections = split_by_markdown_headers(text)
        assert len(sections) == 4
        assert sections[0][2] == 1
        assert sections[1][2] == 2
        assert sections[2][2] == 3
        assert sections[3][2] == 4

    def test_empty_sections(self):
        """空内容章节被跳过"""
        text = """# 标题1

# 标题2
有内容

# 标题3

"""

        sections = split_by_markdown_headers(text)
        # 标题1 没有内容，空段落
        # 标题2 有内容
        # 标题3 没有内容
        assert len(sections) >= 1
        # 至少标题2的段落应该在
        found = any(s[0] == "标题2" for s in sections)
        assert found

    def test_content_with_markdown_table(self):
        """包含 Markdown 表格的章节"""
        text = """# 风险评估
以下为风险评估表：

| 危害 | 严重度 | 概率 |
|------|--------|------|
| 机械危害 | 高 | 中 |
| 化学危害 | 中 | 低 |

## 风险控制
控制措施如下。"""

        sections = split_by_markdown_headers(text)
        assert len(sections) == 2
        assert sections[0][0] == "风险评估"
        assert "机械危害" in sections[0][1]
        assert sections[1][0] == "风险控制"

    def test_mixed_heading_levels(self):
        """混合标题层级"""
        text = """文档开头内容，没有标题

# 风险管理计划
计划内容

### 风险可接受准则
准则内容

## 危害分析
分析内容"""

        sections = split_by_markdown_headers(text)
        assert len(sections) == 4
        assert sections[0][0] == "文档开头"
        assert sections[1][0] == "风险管理计划"
        assert sections[1][2] == 1
        assert sections[2][0] == "风险可接受准则"
        assert sections[2][2] == 3
        assert sections[3][0] == "危害分析"
        assert sections[3][2] == 2

    def test_empty_input(self):
        """空输入"""
        sections = split_by_markdown_headers("")
        assert len(sections) == 0

    def test_whitespace_only_input(self):
        """只有空白的输入"""
        sections = split_by_markdown_headers("   \n\n   \n   ")
        assert len(sections) == 0

    def test_heading_without_space(self):
        """标题#后没有空格，不应识别为标题"""
        text = """#这不是标题
这是正文内容。"""

        sections = split_by_markdown_headers(text)
        # #后无空格，整个文档作为一个段落
        assert len(sections) == 1
        assert sections[0][2] == 0  # level 0 = no heading detected

    def test_chinese_numbered_headings_in_markdown(self):
        """中文编号在标题文本中"""
        text = """# 一、风险管理计划
计划内容

# 二、危害识别
识别内容

## 2.1 物理危害
物理危害内容"""

        sections = split_by_markdown_headers(text)
        assert len(sections) == 3
        assert sections[0][0] == "一、风险管理计划"
        assert sections[1][0] == "二、危害识别"
        assert sections[2][0] == "2.1 物理危害"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
