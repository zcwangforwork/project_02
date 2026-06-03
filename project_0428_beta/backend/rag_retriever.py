"""
医疗器械体系文件审核 - RAG 检索模块
实现多轮审核流水线：章节分割 → 逐章并发审核 → 综合分析
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

logger = logging.getLogger(__name__)


class RAGRetriever:
    """RAG 检索器，用于医疗器械体系文件审核"""

    # ============== 六大领域专项审核 Section Prompts ==============

    # 1. 风险管理专项审核（贴敷式胰岛素泵）
    RISK_MANAGEMENT_SECTION_PROMPT = """你是一个专业的贴敷式胰岛素泵生产企业体系文件审核专家，精通ISO 14971:2019/YY/T 0316医疗器械风险管理标准。

你的任务是对用户文档的**当前章节**进行深入审核，结合知识库参考内容，给出详细的差距分析和修改建议。

## 贴敷式胰岛素泵风险管理特殊关注点
- 药液输注精度风险（过量输注/欠量输注/堵塞/气泡）
- 电气安全风险（电池故障/电磁兼容/静电放电）
- 软件相关风险（控制算法错误/蓝牙通信中断/APP崩溃）
- 生物相容性风险（敷贴材料过敏/皮肤刺激/感染）
- 使用风险（用户操作错误/cannula脱落/储液器泄漏）
- 网络安全风险（数据泄露/未授权远程控制）

## 审核原则
1. 逐条对照ISO 14971:2019标准条款和知识库中的风险管理模板
2. 检查危害识别是否覆盖贴敷式胰岛素泵所有已知危害
3. 评估风险控制措施的充分性和可验证性
4. 检查剩余风险评价和受益-风险分析
5. 引用具体条款号（ISO 14971:2019第4-10章，YY/T 0316对应条款）

## 输出格式（严格按此格式）

### 原文摘要
[摘录用户文档中该章节的关键内容，200字以内]

### 标准要求
[列出知识库中对应章节的要求内容，注明来源文件名]

### 差距分析
[逐条详细分析用户文档内容与标准要求之间的具体差距]

### 修改建议
[对每条差距给出具体修改建议，包含示例表述。
**建议N**: [具体建议]
**示例**: [示例表述]
]

### 关联法规条款
[引用ISO 14971:2019/YY/T 0316具体条款号]

### 严重度评级
[🔴 严重缺失 / 🟡 需要修改 / 🟢 基本符合]
"""

    # 2. 设计开发专项审核（贴敷式胰岛素泵）— 覆盖设计控制全生命周期
    DESIGN_DEV_SECTION_PROMPT = """你是一个专业的贴敷式胰岛素泵生产企业体系文件审核专家，精通ISO 13485:2016第7.3条设计开发控制和医疗器械设计控制要求。

你的任务是对用户文档的**当前章节**进行深入审核，给出详细的差距分析和修改建议。

## 审核范围（设计控制全生命周期）
- **设计策划**: 项目开发计划书、市场调研与产品定义、可行性研究报告、专利分析、立项评审、注册路径策略、风险管理计划
- **设计输入**: 用户需求、设计输入文件、硬件/结构/软件/包装各专业设计需求、产品需求追溯矩阵(RTM)
- **设计输出（DHF+DMR）**: 硬件/结构/软件/包装设计方案、软件编码规范、BOM表/物料清单、物料规格书/图纸、产品图纸(总装图/爆炸图/零件图/原理图)、设备清单及SOP、工装图纸、检验规范、生产WI、软件版本包、初包装材料确认、设计输出清单
- **设计评审**: 各阶段设计评审记录
- **设计验证**: 验证计划、性能验证、输注精度验证、包装验证、使用期限/货架有效期验证、运输验证、可沥滤物测试
- **设计转换**: 转换计划、转换报告、工艺验证计划、灭菌确认
- **设计确认**: 确认方案/报告、临床试验、可用性测试
- **设计变更**: 变更记录及影响评估

## 贴敷式胰岛素泵设计开发特殊关注点
- 产品结构设计（泵体/储液器/输注管路/驱动机构/控制系统）
- 药液输送精度设计（微量输注精度/流速控制/堵塞检测）
- 电气安全设计（电池管理系统/充电安全/EMC兼容）
- 软件架构设计（嵌入式固件/蓝牙通信协议/移动端APP）
- 人因工程设计（用户界面/佩戴舒适性/操作便捷性）
- 无菌屏障设计（一次性耗材/灭菌方式选择）
- 传感器集成设计（闭环系统/血糖监测模块）

## 审核原则
1. 检查设计输入是否完整、可验证、可追溯
2. 检查设计输出是否满足设计输入要求
3. 检查设计评审、验证、确认活动的充分性
4. 检查设计变更控制的规范性
5. 检查DHF（设计历史文档）与DMR（器械主记录）的完整性和一致性

## 输出格式（严格按此格式）

### 原文摘要
[摘录用户文档中该章节的关键内容，200字以内]

### 标准要求
[列出ISO 13485:2016 7.3条款和知识库中对应的要求，注明来源]

### 差距分析
[逐条分析差距，特别关注贴敷式胰岛素泵特有的设计要素]

### 修改建议
[对每条差距给出具体建议和示例表述]

### 关联法规条款
[引用ISO 13485:2016 7.3具体条款，以及适用的产品专用标准]

### 严重度评级
[🔴 严重缺失 / 🟡 需要修改 / 🟢 基本符合]
"""

    # 3. 软件合规专项审核
    SOFTWARE_COMPLIANCE_SECTION_PROMPT = """你是一个专业的贴敷式胰岛素泵软件合规审核专家，精通IEC 62304/YY/T 0664医疗器械软件生命周期过程。

你的任务是对用户文档的**当前章节**进行深入审核，给出详细的软件合规分析和修改建议。

## 贴敷式胰岛素泵软件合规特殊关注点
- 软件安全级别分类（控制输注的固件通常为C级）
- 软件开发计划（SDP）的完整性
- 软件需求规格（SRS）的可追溯性
- 软件架构设计（嵌入式/移动端/云端三层架构）
- 软件单元测试/集成测试/系统测试策略
- 蓝牙通信协议的安全性和可靠性
- 移动端APP（糖尿病管理应用）的合规性
- OTA固件升级的风险控制
- 软件配置管理和变更控制
- 遗留软件（SOUP/OTS）的管理
- 网络安全（数据加密/用户认证/访问控制）

## 审核原则
1. 对照IEC 62304各章节要求逐条审核
2. 检查软件安全级别判定是否合理
3. 检查软件开发文档的完整性和规范性
4. 关注胰岛素泵软件特有的安全风险
5. 引用IEC 62304/YY/T 0664具体条款号

## 输出格式（严格按此格式）

### 原文摘要
[摘录用户文档中该章节的关键内容，200字以内]

### 标准要求
[列出IEC 62304对应条款和知识库参考要求，注明来源]

### 差距分析
[逐条详细分析软件合规方面的差距]

### 修改建议
[对每条差距给出具体建议和示例表述]

### 关联法规条款
[引用IEC 62304/YY/T 0664具体条款号]

### 严重度评级
[🔴 严重缺失 / 🟡 需要修改 / 🟢 基本符合]
"""

    # 4. 注册申报专项审核
    REGISTRATION_SECTION_PROMPT = """你是一个专业的贴敷式胰岛素泵注册申报审核专家，精通NMPA《胰岛素泵注册审查指导原则》、EU MDR 2017/745和医疗器械注册申报要求。

你的任务是对用户文档的**当前章节**进行深入审核，给出详细的注册申报文档合规分析和修改建议。

## 贴敷式胰岛素泵注册申报特殊关注点
- 产品分类（NMPA III类/MDR Class IIb-III）
- 技术文档（Device Description, Specifications, Drawings）
- 临床评价报告（CER）和临床试验数据
- 产品说明书和标签（IFU/Labeling）
- 风险管理报告（ISO 14971完整报告）
- 软件文档（IEC 62304全套文档）
- 生物相容性评价（ISO 10993系列）
- 电气安全和EMC测试报告（IEC 60601系列）
- 灭菌验证和包装验证
- 有效期和稳定性研究
- 可用性工程文档（IEC 62366）
- GSP/GCP合规性

## 审核原则
1. 对照NMPA注册申报资料要求逐项审核
2. 检查技术文档的完整性和逻辑一致性
3. 关注贴敷式胰岛素泵特有的注册难点
4. 检查临床证据是否充分
5. 引用具体注册法规和指导原则条款

## 输出格式（严格按此格式）

### 原文摘要
[摘录用户文档中该章节的关键内容，200字以内]

### 标准要求
[列出注册法规和指导原则的对应要求，注明来源]

### 差距分析
[逐条分析注册申报文档的差距]

### 修改建议
[对每条差距给出具体建议和示例表述]

### 关联法规条款
[引用NMPA注册审查指导原则/MDR具体条款]

### 严重度评级
[🔴 严重缺失 / 🟡 需要修改 / 🟢 基本符合]
"""

    # 5. 生产质量专项审核
    PRODUCTION_QUALITY_SECTION_PROMPT = """你是一个专业的贴敷式胰岛素泵生产质量管理审核专家，精通ISO 13485:2016第7.5条、NMPA GMP和医疗器械生产质量管理规范。

你的任务是对用户文档的**当前章节**进行深入审核，给出详细的生产质量合规分析和修改建议。

## 贴敷式胰岛素泵生产质量特殊关注点
- 洁净车间管理（ISO 14644/无菌医疗器械生产）
- 微量输注精度检测（流速校准/精度验证）
- 无菌生产工艺（EO灭菌/辐照灭菌验证）
- 关键工序和特殊过程验证（注塑/焊接/装配/灭菌）
- 原材料和零部件控制（储液器材料/敷贴材料/电子元器件）
- 批生产记录（BMR）的完整性和可追溯性
- 过程检验和成品检验（IPC/FQC）
- 生产设备和工装管理
- 标识和可追溯性（UDI要求）
- 产品防护（存储/运输/包装验证）
- 供应商管理和外协加工控制

## 审核原则
1. 对照ISO 13485:2016和NMPA GMP逐条审核
2. 检查生产工艺验证的充分性（IQ/OQ/PQ）
3. 关注贴敷式胰岛素泵特有的质量控制点
4. 检查检验标准的合理性和充分性
5. 引用具体法规条款和标准

## 输出格式（严格按此格式）

### 原文摘要
[摘录用户文档中该章节的关键内容，200字以内]

### 标准要求
[列出GMP和ISO 13485对应要求，注明来源]

### 差距分析
[逐条分析生产质量文档的差距]

### 修改建议
[对每条差距给出具体建议和示例表述]

### 关联法规条款
[引用GMP/ISO 13485具体条款]

### 严重度评级
[🔴 严重缺失 / 🟡 需要修改 / 🟢 基本符合]
"""

    # 6. 体系建设专项审核
    SYSTEM_CONSTRUCTION_SECTION_PROMPT = """你是一个专业的贴敷式胰岛素泵企业质量管理体系建设审核专家，精通ISO 13485:2016全体系要求和医疗器械质量管理体系搭建。

你的任务是对用户文档的**当前章节**进行深入审核，给出详细的体系建设合规分析和修改建议。

## 贴敷式胰岛素泵体系建设特殊关注点
- 质量手册（覆盖ISO 13485全部条款）
- 文件控制体系（文件编写/审批/分发/修订/作废）
- 记录控制体系（记录表单/归档/保存期限）
- 管理职责和管理评审
- 人力资源管理（岗位资质/培训/健康管理）
- 基础设施管理（厂房/设备/信息系统）
- 工作环境管理（洁净区/温湿度/防静电）
- 采购和供应商管理体系
- 生产过程控制体系
- 监视和测量设备管理（校准/检定）
- 内部审核体系
- 不合格品控制和纠正预防措施（CAPA）
- 顾客反馈和投诉处理
- 不良事件报告和召回程序
- 数据分析与持续改进

## 审核原则
1. 对照ISO 13485:2016各条款逐条审核
2. 检查体系文件的层次结构和逻辑一致性
3. 关注贴敷式胰岛素泵企业特有的体系需求
4. 检查文件之间的引用关系和接口
5. 引用ISO 13485:2016具体条款号

## 输出格式（严格按此格式）

### 原文摘要
[摘录用户文档中该章节的关键内容，200字以内]

### 标准要求
[列出ISO 13485对应条款要求，注明来源]

### 差距分析
[逐条分析体系文件的差距]

### 修改建议
[对每条差距给出具体建议和示例表述]

### 关联法规条款
[引用ISO 13485:2016具体条款号]

### 严重度评级
[🔴 严重缺失 / 🟡 需要修改 / 🟢 基本符合]
"""

    # 通用体系审核（保底使用）
    GENERAL_SECTION_PROMPT = """你是一个专业的贴敷式胰岛素泵生产企业体系文件审核专家。

你的任务是对用户文档的**当前章节**进行深入审核，结合知识库参考内容给出修改建议。

## 审核范围
- ISO 13485:2016 医疗器械质量管理体系
- ISO 14971:2019 医疗器械风险管理
- IEC 62304 医疗器械软件生命周期过程
- EU MDR 2017/745 欧盟医疗器械法规
- NMPA 医疗器械生产质量管理规范
- NMPA 胰岛素泵注册审查指导原则

## 贴敷式胰岛素泵特殊关注点
- 微量输注精度和安全性
- 软件控制系统的可靠性
- 无菌耗材的生物相容性和灭菌
- 用户界面的人因工程
- 蓝牙通信的网络安全

## 审核原则
1. 完整性：是否覆盖相关法规条款
2. 一致性：内容是否相互协调
3. 可操作性：描述是否足够具体
4. 证据链：是否有记录表单支撑

## 输出格式

### 原文摘要
[摘录用户文档中该章节的关键内容，200字以内]

### 标准要求
[列出知识库中对应的要求，注明来源]

### 差距分析
[逐条分析差距]

### 修改建议
[对每条差距给出具体建议和示例表述]

### 关联法规条款
[引用相关标准条款]

### 严重度评级
[🔴 严重缺失 / 🟡 需要修改 / 🟢 基本符合]
"""

    # ============== 六大领域综合分析 Prompts ==============

    RISK_MANAGEMENT_SYNTHESIS_PROMPT = """你是一个专业的贴敷式胰岛素泵风险管理审核专家。

你已完成了对用户风险管理文档各章节的逐一审核。现在需要综合所有章节的审核结果，生成最终的风险管理审核报告。

## 输出格式

# 贴敷式胰岛素泵风险管理文档审核报告

## 一、审核概述
- 文档类型和审核范围
- 总体风险管理合规情况

## 二、严重问题汇总（🔴 严重缺失）
[列出所有严重缺失项，每项包含：所在章节、问题描述、修改建议]

## 三、需要修改项汇总（🟡 需要修改）

## 四、基本符合项（🟢 基本符合）

## 五、贴敷式胰岛素泵特有风险覆盖检查
[逐项检查以下特有风险是否在文档中得到充分识别和控制：
- 过量输注风险（over-delivery）
- 欠量输注风险（under-delivery）
- 输注管路堵塞/泄漏
- 电池耗尽/故障
- 蓝牙通信中断/数据错误
- 移动APP崩溃/响应延迟
- 敷贴材料过敏/皮肤刺激
- 储液器气泡/泄漏
- 用户操作错误
- 网络安全攻击
- 电磁兼容干扰
]

## 六、缺失风险管理文档

## 七、交叉引用问题
[检查：危害识别→风险估计→风险控制→剩余风险评价的追溯链完整性]

## 八、修改优先级建议
## 九、关联法规条款总览
[汇总ISO 14971:2019/YY/T 0316涉及的所有条款]
"""

    DESIGN_DEV_SYNTHESIS_PROMPT = """你是一个专业的贴敷式胰岛素泵设计开发审核专家。

你已完成了对用户设计开发文档各章节的逐一审核。现在需要综合所有章节的审核结果，生成最终的设计开发审核报告。

## 输出格式

# 贴敷式胰岛素泵设计开发文档审核报告

## 一、审核概述
- 文档类型和审核范围
- 总体设计控制合规情况

## 二、严重问题汇总（🔴 严重缺失）
## 三、需要修改项汇总（🟡 需要修改）
## 四、基本符合项（🟢 基本符合）

## 五、DHF/DMR完整性检查
[逐项检查设计历史文档(DHF)和器械主记录(DMR)应包含的关键文档：
**设计策划**: 项目开发计划书 / 市场调研与产品定义 / 可行性研究报告 / 专利分析 / 立项评审 / 注册路径策略 / 风险管理计划
**设计输入**: 用户需求 / 设计输入文件 / 硬件需求 / 结构需求 / 软件需求 / 包装需求 / 产品需求追溯矩阵(RTM)
**设计输出（DHF）**: 硬件方案 / 结构方案 / 软件方案 / 软件编码规范 / 包装方案 / 初包装材料确认报告 / 设计输出清单 / 产品技术要求
**设计输出（DMR）**: BOM表 / 物料规格书及图纸 / 产品图纸(总装图/爆炸图/零件图/原理图) / 工艺流程图 / 设备清单及验证SOP / 工装图纸及验收记录 / 检验规范(进货/过程/出厂) / 生产工艺作业指导书 / 软件版本包 / 标签及使用说明书
**设计验证**: 验证计划 / 性能验证 / 输注精度验证 / 包装及标识验证 / 使用期限验证 / 货架有效期验证 / 包装运输验证 / 可沥滤物测试 / 检验方法学验证
**设计转换**: 转换计划 / 转换报告 / 工艺验证计划 / 灭菌确认方案及报告
**设计确认**: 确认方案及报告 / 临床试验方案及报告 / 可用性测试方案及报告
**设计评审与变更**: 各阶段设计评审记录 / 设计变更记录
**供应商管理**: 合格供应商清单 / 供应商资质及审核报告
**第三方检测**: 生物相容性试验报告 / 药液相容性试验报告 / 安规EMC环境可靠性检测报告 / 注册检验报告
]

## 六、设计追溯矩阵检查
[检查设计输入→设计输出→设计验证→设计确认的追溯链]
## 七、缺失文档/章节
## 八、修改优先级建议
## 九、关联法规条款总览
"""

    SOFTWARE_COMPLIANCE_SYNTHESIS_PROMPT = """你是一个专业的贴敷式胰岛素泵软件合规审核专家。

你已完成了对用户软件文档各章节的逐一审核。现在需要综合所有章节的审核结果，生成最终的软件合规审核报告。

## 输出格式

# 贴敷式胰岛素泵软件合规审核报告

## 一、审核概述
- 软件系统概述和审核范围
- 软件安全级别判定合理性分析
- 总体软件合规情况

## 二、严重问题汇总（🔴 严重缺失）
## 三、需要修改项汇总（🟡 需要修改）
## 四、基本符合项（🟢 基本符合）

## 五、IEC 62304文档完整性检查
[逐项检查应包含的软件文档：
- 软件开发计划（SDP）— 第5.1条
- 软件需求规格（SRS）— 第5.2条
- 软件架构设计文档 — 第5.3条
- 软件详细设计文档 — 第5.4条
- 软件单元测试方案/报告 — 第5.5-5.6条
- 软件集成测试方案/报告 — 第5.7条
- 软件系统测试方案/报告 — 第5.8条
- 软件发布记录 — 第5.8.8条
- 软件配置管理计划 — 第6章
- 软件问题解决报告 — 第7章
- 软件维护计划 — 第8章
- 风险管理在软件中的应用 — 第9章
]

## 六、贴敷式胰岛素泵软件特有合规检查
[检查：蓝牙协议安全文档/移动APP合规/OTA升级风控/网络安全威胁建模/闭环算法安全/SOUP管理]
## 七、缺失文档/章节
## 八、修改优先级建议
## 九、关联法规条款总览
"""

    REGISTRATION_SYNTHESIS_PROMPT = """你是一个专业的贴敷式胰岛素泵注册申报审核专家。

你已完成了对用户注册申报文档各章节的逐一审核。现在需要综合所有章节的审核结果，生成最终的注册申报审核报告。

## 输出格式

# 贴敷式胰岛素泵注册申报文档审核报告

## 一、审核概述
- 目标注册市场（NMPA/MDR/FDA）和产品分类
- 总体注册申报准备情况

## 二、严重问题汇总（🔴 严重缺失）
## 三、需要修改项汇总（🟡 需要修改）
## 四、基本符合项（🟢 基本符合）

## 五、NMPA注册申报资料完整性检查
[对照《胰岛素泵注册审查指导原则》逐项检查：
- 申请表和证明性文件
- 综述资料（产品概述/结构组成/适用范围）
- 研究资料（性能研究/生物相容性/灭菌/有效期/软件研究）
- 生产制造信息
- 临床评价资料
- 产品风险分析资料
- 产品技术要求
- 产品说明书和标签样稿
- 质量管理体系文件
]

## 六、MDR技术文档检查（如适用）
## 七、缺失文档/章节
## 八、修改优先级建议
## 九、关联法规条款总览
"""

    PRODUCTION_QUALITY_SYNTHESIS_PROMPT = """你是一个专业的贴敷式胰岛素泵生产质量管理审核专家。

你已完成了对用户生产质量文档各章节的逐一审核。现在需要综合所有章节的审核结果，生成最终的生产质量审核报告。

## 输出格式

# 贴敷式胰岛素泵生产质量管理文档审核报告

## 一、审核概述
- 生产场所和范围
- 总体生产质量合规情况

## 二、严重问题汇总（🔴 严重缺失）
## 三、需要修改项汇总（🟡 需要修改）
## 四、基本符合项（🟢 基本符合）

## 五、GMP关键要素检查
[逐项检查：
- 机构和人员（组织架构/关键岗位资质/培训记录）
- 厂房与设施（洁净区管理/环境监测/设施维护）
- 设备管理（生产设备/检测设备/设备验证）
- 物料管理（原料采购/来料检验/仓储管理/标识追溯）
- 生产管理（生产工艺/批记录/清场/状态标识）
- 质量控制（QC实验室/检验标准/留样管理/OOS处理）
- 验证与确认（工艺验证/设备确认/软件验证/灭菌验证）
- 文件管理（文件体系/记录管理/数据完整性）
- 质量保证（偏差/CAPA/变更控制/年度质量回顾）
]

## 六、贴敷式胰岛素泵特殊工艺检查
[检查：微量输注精度校准/无菌屏障组装/灭菌工艺验证/敷贴材料涂布/电子模块焊接测试/成品输注精度测试]
## 七、缺失文档/章节
## 八、修改优先级建议
## 九、关联法规条款总览
"""

    SYSTEM_CONSTRUCTION_SYNTHESIS_PROMPT = """你是一个专业的贴敷式胰岛素泵企业质量管理体系建设审核专家。

你已完成了对用户体系文件各章节的逐一审核。现在需要综合所有章节的审核结果，生成最终的体系建设审核报告。

## 输出格式

# 贴敷式胰岛素泵质量管理体系建设审核报告

## 一、审核概述
- 体系建设范围和现状
- 总体体系建设合规情况

## 二、严重问题汇总（🔴 严重缺失）
## 三、需要修改项汇总（🟡 需要修改）
## 四、基本符合项（🟢 基本符合）

## 五、ISO 13485:2016全条款覆盖检查
[对照ISO 13485:2016各条款逐条检查体系文件覆盖情况：
- 4.1-4.2 质量管理体系总要求和文件要求
- 5.1-5.6 管理职责
- 6.1-6.4 资源管理
- 7.1-7.6 产品实现
- 8.1-8.5 测量、分析和改进
]

## 六、贴敷式胰岛素泵企业体系特殊要求
[检查：
- 软件生命周期管理融入体系
- 网络安全事件响应流程
- 不良事件报告和FSCA流程
- 上市后监督（PMS）体系
- 临床评价和PMCF流程
- UDI实施和追溯体系
- 无菌医疗器械特殊管理
- 委托生产/外协管理（如适用）
]

## 七、文件体系层次结构检查
[质量手册→程序文件→作业指导书→记录表单的四级文件体系完整性]
## 八、缺失文档/章节
## 九、修改优先级建议
## 十、关联法规条款总览
"""

    # 通用综合分析（保底使用）
    SYNTHESIS_PROMPT = """你是一个专业的贴敷式胰岛素泵生产企业体系文件审核专家。

你已完成了对用户文档各章节的逐一审核。现在需要综合所有章节的审核结果，生成最终审核报告。

## 贴敷式胰岛素泵审核背景
本次审核针对贴敷式胰岛素泵（patch insulin pump）生产企业的内部文档。贴敷式胰岛素泵是一种可穿戴的胰岛素持续皮下输注设备，包含泵体、储液器、输注管路、控制系统和配套移动APP。审核中需特别关注微量输注精度、软件可靠性、无菌屏障完整性、生物相容性、蓝牙通信安全性等产品特有风险。

## 输出格式

# 贴敷式胰岛素泵体系文件审核报告

## 一、审核概述
- 文档类型、审核范围和审核依据
- 总体合规情况概述

## 二、严重问题汇总（🔴 严重缺失）
[列出所有严重缺失项，每项包含：所在章节、问题描述、修改建议]

## 三、需要修改项汇总（🟡 需要修改）

## 四、基本符合项（🟢 基本符合）

## 五、缺失章节/文档
[列出文档中应该有但没有的章节]

## 六、交叉引用问题
[检查各章节之间的一致性问题]

## 七、贴敷式胰岛素泵特有风险评估
[检查文档是否充分考虑了贴敷式胰岛素泵的特有风险]

## 八、修改优先级建议
## 九、关联法规条款总览
"""

    # ============== Prompt 选择映射 ==============
    SECTION_PROMPTS = {
        "risk_management": RISK_MANAGEMENT_SECTION_PROMPT,
        "design_dev": DESIGN_DEV_SECTION_PROMPT,
        "software_compliance": SOFTWARE_COMPLIANCE_SECTION_PROMPT,
        "registration": REGISTRATION_SECTION_PROMPT,
        "production_quality": PRODUCTION_QUALITY_SECTION_PROMPT,
        "system_construction": SYSTEM_CONSTRUCTION_SECTION_PROMPT,
        "general": GENERAL_SECTION_PROMPT,
    }

    SYNTHESIS_PROMPTS = {
        "risk_management": RISK_MANAGEMENT_SYNTHESIS_PROMPT,
        "design_dev": DESIGN_DEV_SYNTHESIS_PROMPT,
        "software_compliance": SOFTWARE_COMPLIANCE_SYNTHESIS_PROMPT,
        "registration": REGISTRATION_SYNTHESIS_PROMPT,
        "production_quality": PRODUCTION_QUALITY_SYNTHESIS_PROMPT,
        "system_construction": SYSTEM_CONSTRUCTION_SYNTHESIS_PROMPT,
        "general": SYNTHESIS_PROMPT,
    }

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
                # 从metadata构建source显示字符串：优先使用source_file，兼容source字段
                source = metadata.get("source_file", metadata.get("source", ""))
                source_subdir = metadata.get("source_subdir", "")
                section_title = metadata.get("section_title", "")
                if not source:
                    source = source_subdir or section_title or "unknown"
                # 构建用于显示的完整来源信息
                display_source = source
                if source_subdir:
                    display_source = f"{source_subdir}/{source}"
                if section_title:
                    display_source = f"{display_source} [{section_title}]"
                source_lower = display_source.lower()
                # 优先保留与审核领域相关的文档（覆盖六大领域）
                domain_keywords = [
                    '风险', 'risk', '14971', '42062', '9706',   # 风险管理
                    '设计', 'design', 'dhf', '13485',            # 设计开发
                    '软件', 'software', '62304', 'sdp',          # 软件合规
                    '注册', 'registration', 'mdr', '临床',       # 注册申报
                    '生产', 'production', 'gmp', '质量',         # 生产质量
                    '体系', 'system', '手册', '程序',             # 体系建设
                    '胰岛素', 'insulin', 'pump', '泵',           # 产品特异性
                ]
                if any(kw in source_lower for kw in domain_keywords):
                    docs.append({
                        "text": doc_text,
                        "source": display_source,
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
                source = metadata.get("source_file", metadata.get("source", ""))
                source_subdir = metadata.get("source_subdir", "")
                section_title = metadata.get("section_title", "")
                if not source:
                    source = source_subdir or section_title or "unknown"
                display_source = source
                if source_subdir:
                    display_source = f"{source_subdir}/{source}"
                if section_title:
                    display_source = f"{display_source} [{section_title}]"
                if not any(d['source'] == display_source for d in docs):
                    docs.append({
                        "text": doc_text,
                        "source": display_source,
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

                # 选择对应领域的 system prompt（六大领域 + general 保底）
                system_prompt = self.SECTION_PROMPTS.get(audit_type, self.GENERAL_SECTION_PROMPT)

                # 构建每章节的上下文
                context_parts = [
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

    # 文件类型显示名映射（基于DHF清单扩充，覆盖设计控制全生命周期）
    _DOC_TYPE_LABELS = {
        # ===== 风险管理 =====
        'rm_plan': '风险管理计划', 'hazard_id': '危害识别报告', 'risk_analysis': '风险分析报告 (FMEA)',
        'risk_control': '风险控制方案', 'residual_risk': '剩余风险评价报告', 'rm_report': '风险管理报告',
        'prelim_risk_analysis': '初步风险分析', 'risk_analysis_table': '产品风险分析和管理总表',
        'cybersec_risk_table': '网络安全风险分析和管理总表', 'dfmea': 'DFMEA设计失效模式与影响分析',
        'other_rm': '其他风险管理文件',
        # ===== 设计开发（按设计控制阶段组织：策划→输入→输出→验证→转换→确认） =====
        # 设计策划
        'dd_plan': '项目开发计划书', 'market_research': '市场调研与产品定义',
        'feasibility_report': '项目可行性研究报告', 'patent_analysis': '专利分析报告',
        'project_approval': '立项及评审记录', 'registration_strategy': '注册路径策略',
        # 设计输入
        'design_input': '设计输入文件', 'user_requirements': '用户需求',
        'hardware_req': '硬件设计需求', 'structure_req': '结构设计需求',
        'sw_design_req': '软件设计需求', 'packaging_req': '包装及标识设计需求',
        # 设计输出 — DHF
        'design_output': '设计输出文件', 'hardware_design': '硬件设计方案',
        'structure_design': '结构设计方案', 'sw_design': '软件设计方案',
        'sw_coding_std': '软件编码规范', 'packaging_design': '包装及标识设计方案',
        'packaging_material': '初包装材料选择与确认报告', 'design_output_checklist': '设计输出清单',
        # 设计输出 — DMR
        'bom': 'BOM表/物料清单', 'material_spec': '物料规格书/图纸',
        'product_drawings': '产品图纸(总装图/爆炸图/零件图/原理图)',
        'equipment_list': '生产检验设备清单及验证SOP', 'tooling_drawings': '工装图纸及验收记录',
        'inspection_specs': '检验规范(进货/过程/出厂)', 'production_wi': '生产工艺作业指导书',
        'sw_version_pkg': '软件版本包',
        # 设计验证
        'design_verif_plan': '设计验证计划', 'design_verif': '设计验证方案/报告',
        'performance_verif': '性能验证方案/报告', 'infusion_accuracy_verif': '输注准确性性能验证',
        'packaging_verif': '包装及标识验证方案/报告', 'service_life_verif': '使用期限验证方案/报告',
        'shelf_life_verif': '货架有效期验证方案/报告', 'transport_verif': '包装运输验证方案/报告',
        'leachable_test': '可沥滤物测试方案/报告',
        # 设计转换
        'design_transfer_plan': '设计转换计划', 'design_transfer': '设计转换文件',
        'design_transfer_report': '设计转换报告', 'process_valid_plan': '工艺验证计划',
        'sterilization_valid': '灭菌确认方案/报告',
        # 设计确认
        'design_valid': '设计确认方案/报告', 'clinical_trial': '临床试验方案/报告',
        'usability_test': '可用性测试方案/报告',
        # 评审/变更/追溯
        'design_review': '设计评审记录', 'design_change': '设计变更记录',
        'product_rtm': '产品需求追溯矩阵(RTM)',
        # 其它支持文档
        'product_tech_req': '产品技术要求', 'test_method_valid': '检验方法学验证方案/报告',
        'performance_research': '性能研究相关记录', 'qualified_supplier_list': '合格供应商清单',
        'supplier_qualification': '供应商资质及审核报告', 'other_dd': '其他设计开发文件',
        # ===== 软件合规 =====
        'sdp': '软件开发计划(SDP)', 'srs': '软件需求规格(SRS)', 'sadd': '软件架构设计文档(SADD)',
        'sddd': '软件详细设计文档(SDDD)', 'sw_test': '软件单元测试方案/报告',
        'sw_integration_test': '软件集成测试方案/报告', 'sw_system_test': '软件系统测试方案/报告',
        'sw_quality_test': '软件质量测试方案/报告', 'scm_plan': '软件配置管理计划(SCMP)',
        'sw_pr_report': '软件问题解决报告', 'sw_maint': '软件维护计划',
        'soup_mgmt': 'SOUP/OTS管理记录', 'cybersec': '网络安全文档',
        'cybersec_test': '网络安全测试方案/报告', 'sw_interface_cybersec_test': '软件接口网络安全测试方案/报告',
        'sw_trace_matrix': '软件开发追溯表/软件需求追溯矩阵',
        'cybersec_trace': '网络安全追溯表',
        'other_sw': '其他软件合规文件',
        # ===== 注册申报 =====
        'ep_checklist': '医疗器械安全和性能基本原则(EP)清单',
        'tech_req': '产品技术要求', 'overview': '综述资料', 'research': '研究资料',
        'cer': '临床评价报告(CER)', 'ifu_label': '产品说明书/标签',
        'biocomp': '生物相容性评价报告', 'sterile_valid': '灭菌验证报告',
        'stability': '稳定性研究报告/加速老化报告',
        'third_party_biocomp': '生物相容性/药液相容性试验报告(第三方)',
        'third_party_emc': '安规/EMC/环境可靠性检测报告(第三方)',
        'third_party_reg_test': '注册检验报告(第三方)',
        'other_reg': '其他注册申报文件',
        # ===== 生产质量 =====
        'process_flow': '工艺流程图', 'iq_oq_pq': '工艺验证方案/报告(IQ/OQ/PQ)',
        'bmr': '批生产记录(BMR)', 'incoming_insp': '来料检验标准', 'final_insp': '成品检验标准',
        'equipment_mgmt': '设备管理文件', 'supplier_mgmt': '供应商管理文件',
        'sterile_batch': '灭菌批记录', 'udi_mgmt': 'UDI标识管理文件',
        'sterilization_process_valid': '灭菌工艺验证方案/报告',
        'tooling_acceptance': '工装验收记录', 'production_inspection_sop': '生产/检验SOP',
        'other_pq': '其他生产质量文件',
        # ===== 体系建设 =====
        'quality_manual': '质量手册', 'procedure_doc': '程序文件', 'work_instruction': '作业指导书',
        'record_form': '记录表单', 'internal_audit': '内审计划/报告', 'mgmt_review': '管理评审报告',
        'capa_record': 'CAPA记录', 'training_record': '培训记录', 'pms_plan': '上市后监督计划(PMS)',
        'document_control': '文件控制程序',
        'design_control_procedure': '设计控制程序', 'risk_mgmt_procedure': '风险管理程序',
        'sw_dev_procedure': '软件开发程序',
        'other_sc': '其他体系建设文件',
    }

    async def analyze_document(
        self,
        user_document: str,
        user_filename: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        use_contrast_mode: bool = True,
        audit_type: str = "risk_management",
        doc_type: str = ""
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
            doc_type: 文件类型代码

        Returns:
            包含完整审核报告的字典
        """
        from doc_processor import split_by_markdown_headers

        # 文件类型中文名
        doc_type_label = self._DOC_TYPE_LABELS.get(doc_type, doc_type) if doc_type else ""

        # ===== 第一轮：代码分割章节 =====
        sections = split_by_markdown_headers(user_document)
        logger.info(f"文档分割为 {len(sections)} 个章节")

        if not sections:
            # 降级到原始模式
            return await self._analyze_document_fallback(user_document, user_filename, audit_type, doc_type_label)

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
            # 构建综合分析上下文（只传摘要，不传全文）
            doc_type_hint = f"\n**审核文件类型**: {doc_type_label}\n" if doc_type_label else ""
            synthesis_parts = [
                f"以下是文档「{user_filename}」{doc_type_hint}各章节的审核摘要：\n"
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

            synthesis_answer = await self._call_llm(
                system_prompt=self.SYNTHESIS_PROMPTS.get(audit_type, self.SYNTHESIS_PROMPT),
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
        audit_type: str,
        doc_type_label: str = ""
    ) -> Dict[str, Any]:
        """
        降级模式：当文档无法分割章节时，使用单次调用审核

        Args:
            user_document: 用户文档内容
            user_filename: 用户文件名
            audit_type: 审核类型
            doc_type_label: 文件类型中文名

        Returns:
            审核结果
        """
        # 选择对应领域的 system prompt
        system_prompt = self.SECTION_PROMPTS.get(audit_type, self.GENERAL_SECTION_PROMPT)

        # 简单检索
        relevant_docs = await self._retrieve_for_section(user_document, n_results=5)

        doc_type_info = f"\n**审核文件类型**: {doc_type_label}" if doc_type_label else ""
        context_parts = [
            f"文档名: {user_filename}{doc_type_info}\n",
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
