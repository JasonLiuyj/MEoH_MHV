# 独立版 MEoH（来自 HV Estimation）

本项目从当前 LLM4AD 的 HV Estimation 实验中提取了实际使用的增强版 MEoH，保留最小运行框架，不包含 HV 数据集、历史日志、绘图脚本和三个 HV baseline。

保留的 MEoH 功能包括：

- 原始多目标 E2、M1、M2 算子；
- P1、T1 双方向算子；
- TIDE 参数微调；
- RAISE 困难实例刷新接口；
- PFG 与基于 LLM 的聚类繁殖；
- flash reflection、harmony search 和 resume 支持；
- JSON 种群及精英集日志。

## 直接运行

1. 在 PyCharm 中将本目录作为项目打开。
2. 选择 Python 解释器并安装 `requirements.txt`。
3. 检查项目根目录 `.env` 中的 `LLM4AD_API_HOST`、`LLM4AD_API_KEY`、`LLM4AD_MODEL` 和 `LLM4AD_TIMEOUT`。
4. 右键 `run_meoh.py`，选择 **Run 'run_meoh'**。

入口默认运行 `problem/example_problem.py` 中的小型双目标示例。它只用来验证新问题接入方式，不是 HV 实验。

## 接入新问题

复制 `problem/example_problem.py` 并修改：

1. `TEMPLATE_PROGRAM`：定义希望 LLM 改写的函数，函数名和接口必须固定。
2. `task_description`：用自然语言说明算法目标、输入、输出与约束。
3. `evaluate_program()`：在全部训练实例上调用 `callable_func`，返回目标向量。
4. `run_meoh.py`：导入新 Evaluation，并调整种群、预算、并发和组件开关。

MEoH 对所有目标都执行最大化。绝对误差、运行时间、成本等需要最小化的指标，应返回其负值，例如 `[-mean_error, -mean_runtime]`。无效算法返回 `None`。

启用 RAISE 前，新 Evaluation 还必须实现 `raise_candidates()`、`available_raise_instance_ids()`、`add_raise_instance()` 和 `rollback_raise_instance()`。RAISE 的领域实例生成逻辑不属于通用 MEoH 核心，需要随新问题实现。

## 目录边界

- `llm4ad/method/meoh/`：HV Estimation 使用的增强版 MEoH 核心。
- `llm4ad/base/`：代码表示、LLM 抽象及安全评估器。
- `llm4ad/tools/`：日志 profiler、HTTPS LLM 客户端和 `.env` 读取。
- `problem/`：新领域代码应主要放在这里。
- `run_meoh.py`：可由 PyCharm 直接执行的实验入口。

代码来源于 LLM4AD 项目及当前工作区中的研究扩展。使用和发表时请保留源文件中的 LLM4AD 版权与引用说明。
