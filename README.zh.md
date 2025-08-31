# 投资顾问

## 免责声明：本项目仅用于学习和研究 AI 技术，请勿作为实际投资决策的依据。若因参考本项目进行投资导致任何损失，作者概不负责。投资有风险，入市需谨慎。

## 注意：生成的报告数据可能并非 100%准确，请自行判断。

端到端、可审计的从研究到执行的价值投资、风险优先基金工作流。它会生成：

- 研究备忘录、估值要点和情景分析（熊市/基准/牛市）
- 投资组合建议（目标权重、建仓计划、流动性、约束条件）
- 风险与合规检查（VaR、压力测试、软上限）
- 交易计划（算法、成本估算、TCA 基线）
- 可选：投委会/业绩评审用 Markdown 报告

# 示例

[示例](examples/zh/特斯拉/report_2025-08-31.md)

## 架构总览

- Orchestrator：驱动流程并将产物写入 `knowledge/`
- 研究代理：生成结构化的 ResearchMemo（投资逻辑、护城河、估值、风险）
- 投资经理代理：提出 Proposal（目标权重、建仓计划、流动性、约束条件）
- 风险/合规：校验 VaR/限制及受限名单（MVP）
- 交易代理：选择算法和成本估算
- 报告代理：渲染带图表的 Markdown 报告

产物（JSON）：

- `knowledge/research/<instrument>/memo_<date>.json`
- `knowledge/proposals/<instrument>/proposal_<date>.json`
- `knowledge/risk/<date>/risk_<instrument>.json`
- `knowledge/compliance/<date>/compliance_<instrument>.json`
- `knowledge/trading/<date>/plan_<instrument>.json`
- 可选：`knowledge/proposals/<instrument>/report_<date>.md` 及 `figs/` 下的图表

## 依赖

- Python 3.11+
- OpenAI API 密钥：`OPENAI_API_KEY`（必需）
- 可选 MCP 服务器用于市场数据/搜索：
  - 金融数据集 MCP: `FINANCIAL_DATASETS_API_KEY`
  - 谷歌搜索 MCP: `GOOGLE_API_KEY`，`GOOGLE_SEARCH_ENGINE_ID`

你也可以通过 `OPENAI_MODEL` 配置模型（默认：gpt-4o-mini）。

## 快速开始

1. 创建虚拟环境并安装依赖

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. 设置环境变量

```bash
export OPENAI_API_KEY=sk-...
# 可选模型覆盖
export OPENAI_MODEL=gpt-4o-mini
# 可选MCP密钥（若使用 --use-mcp）
export FINANCIAL_DATASETS_API_KEY=...
export GOOGLE_API_KEY=...
export GOOGLE_SEARCH_ENGINE_ID=...
```

3. 运行流程（仅 LLM 路径）

```bash
python main.py --instrument "Analyze TESLA" --advice --report --lang en
```

输出将写入 `knowledge/` 目录。

## MCP 集成

你可以通过以下初始化脚本引导，克隆 MCP 服务器并生成 `mcp/mcp.json` 指向本地路径。

启用 MCP 运行：

```bash
python main.py --instrument "Analyze TESLA" --use-mcp --advice --report --lang en
```

## 项目初始化（MCP 引导）

提供了便捷脚本用于建立 venv、安装 Python 依赖，并（可选）克隆 MCP 服务器。

```bash
bash scripts/init_project.sh
```

该脚本：

- 创建/激活 `venv` 并安装 `requirements.txt`
- 若提供 URL 则克隆 MCP 服务器至 `mcp/vendors/`
- 若有则构建 Google Search MCP（npm）
- 生成指向本地克隆的 `mcp/mcp.json`

如果你已在其他地方维护 MCP 服务器，可保持 `mcp/mcp.json` 不变。

## 配置（策略）

基线风险限制可通过环境变量配置：

- `POLICY_MAX_SINGLE_NAME`（默认 0.06）
- `POLICY_SECTOR_CAP`（默认 0.20）
- `POLICY_VAR_SOFT_CAP`（默认 0.035）

投资经理代理在生成建议时会强制执行这些限制，并调整建仓进度以实现从现有仓位到目标仓位的过渡。

## 代码结构

- `ai_advisor/schemas.py`：所有产物的 Pydantic schema
- `ai_advisor/agents.py`：研究/投资经理/交易代理和建议归一化
- `ai_advisor/risk_compliance.py`：VaR/限制检查（MVP）
- `ai_advisor/orchestrator.py`：流程和产物写入
- `ai_advisor/report_agent.py`：报告渲染和图表生成
- `ai_advisor/policy.py`：策略基线从环境变量加载
- `main.py`：CLI 入口

## 备注

- 本仓库仅写本地文件，不涉及真实交易/执行。
- 你可以替换模型、提示词、风险策略，或以极小改动引入 MCP 工具。
