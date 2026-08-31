# Kimi K3 智能体上下文资源实验

本仓库将《深入理解 AI Agent》第 1 章实验 1-1 扩展为可审计的资源消融实验。实验固定使用 Kimi K3、受约束多币种任务、原工具实现和五种上下文配置，每种配置独立运行 30 次。

## 实验变量

- `full`：完整上下文
- `no_history`：移除历史
- `no_reasoning`：移除历史推理
- `no_tool_calls`：移除工具定义
- `no_tool_results`：移除工具结果

五种配置按五阶平衡拉丁方串行运行。最大循环轮数为 5，工具结果消融使用空内容。

## 运行

```powershell
uv venv
uv pip install -r resource_experiment/requirements.txt
.\.venv\Scripts\python.exe -m pytest resource_experiment/tests -q
.\.venv\Scripts\python.exe -m resource_experiment.run --phase pilot
.\.venv\Scripts\python.exe -m resource_experiment.run --phase main --repetitions 30 --resume
.\.venv\Scripts\python.exe -m resource_experiment.analyze
.\.venv\Scripts\python.exe -m resource_experiment.validate
```

`MOONSHOT_API_KEY` 仅从进程或 Windows 用户级环境变量读取，不写入仓库。

## 数据产物

- `results/<experiment_id>/events.jsonl`：逐事件原始日志
- `results/<experiment_id>/runs/*.json`：任务级完整记录
- `results/<experiment_id>/derived/`：任务表、配置汇总和配对差值
- `results/<experiment_id>/figures/`：统计图
- `results/<experiment_id>/REPORT.md`：实验报告

验收采用结构解析、字段约束、数量核对、运行编号唯一性和结束事件检查，不生成文件哈希。

## 上游来源

实验基线来自 [`bojieli/ai-agent-book`](https://github.com/bojieli/ai-agent-book)，固定版本为 `030397201cc82d2c5b4d375a9b1ea52b15e28db9`。本地远程 `upstream` 保留该来源。

