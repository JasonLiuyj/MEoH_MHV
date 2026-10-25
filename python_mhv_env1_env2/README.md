# 第一、第二阶段 MHV 曲线实验

本文件夹是从 `python_mhv_experiment` 复制的独立实验，包含算法、问题、MHV 指标和三个 setting 的配置，可单独复制到其他电脑运行。

云服务器使用时，请上传整份代码包，包括 `run_experiments.py`、`algorithms/base.py` 和 `algorithms/dynamic_algorithms.py`。启动新版会输出 `Code version: env1-env2-stage2-fix-v1` 和实际脚本路径。已经启动的进程不会自动加载修改后的代码，更新后需重新启动。每次运行结束应能看到 `Stage 2 Locked` 或此前已记录的该阶段末尾值，以及 `Stage 2 MHV samples=...`。

从第一阶段开始采样，经历第一次环境变化后继续采样，到第二阶段结束时停止。三个 setting 的完整目标序列保留，以保持决策变量维数、全局目标数和 WFG 参数与原实验一致。绘图只显示第一、第二阶段，横轴使用原始 FE/N。

| τ | 第一阶段 | 第二阶段 | 横轴范围 |
|---|---|---|---|
| 25 | [0, 100) | [100, 125) | 0–125 |
| 50 | [0, 100) | [100, 150) | 0–150 |
| 100 | [0, 100) | [100, 200) | 0–200 |

采样包含初始化种群，采用原程序的 `(FE-1)//N` 编号。τ=50 时最后一个阶段内采样最多为 149，横轴右边界为 150。不同算法每轮消耗的评估次数不同，实际采样可能跳过部分 FE/N 点，不补造采样值。若旧阶段种群在一次更新完成时已跨过名义边界，末尾采样使用种群最新的实际评估时间。均值和标准差由同一阶段、同一 FE/N 的实际采样计算；`_summary.csv` 的 `sample_count` 给出贡献的运行次数。

## 运行

Python 3.9 或以上，安装依赖：

```powershell
cd "D:\MATLAB\PlatEMO-master\PlatEMO\Problems\Multi-objective optimization\DTLZ\python_mhv_env1_env2"
python -m pip install -r requirements.txt
```

画第一阶段开始到第二阶段结束的曲线（τ=50），默认五种算法，三个 setting，先用 5 次独立运行：

```powershell
python run_experiments.py --problems DTLZ1 --taus 50 --settings 1,2,3 --runs 5
```

将 `DTLZ1` 改为需要的问题，如 `DTLZ4`，或使用逗号选择多个问题。未指定 `--problems` 时采用原实验的全部十个问题；未指定 `--taus` 时采用 25、50、100。三个 setting 的动态目标序列与原配置一致。

正式实验（原实验的 N=150、31 次运行、十个问题和三个 τ）：

```powershell
python run_experiments.py
```

该命令运行规模较大。高维 MHV 保留原指标的计算精度；`--pf-count` 不会降低已支持问题的解析 PF-HV 精度或高维 HV 的蒙特卡洛采样量。

默认算法及图例对应关系：

| 参数中的名称 | 图例 |
|---|---|
| DTAEA | DTAEA |
| KTDMOEA | KTDMOEA |
| LEC | LEC |
| STA | STA |
| TrustArchiveSPEA2SDE | RAAT-SPEA2SDE |

`TrustSPEA2SDE` 对应代码中的 `TrustArchiveSPEA2SDE`。默认只运行上表五个算法，无需额外指定 `--algorithms`。也可显式指定：

```powershell
python run_experiments.py --problems DTLZ1 --taus 50 --runs 5 --algorithms DTAEA,KTDMOEA,LEC,STA,TrustArchiveSPEA2SDE
```

原完整实验还包含 `RestartNSGAII`，需要时可通过 `--algorithms` 加入。新实验排除不支持曲线采样的 `RandomSearch`。

较快的流程验证（不是正式比较结果）：

```powershell
python run_experiments.py --problems DTLZ2 --taus 50 --settings 1,2,3 --runs 1 --n 10 --algorithms RestartNSGAII --sample-every 10 --output-dir smoke_results
```

## 输出与重复绘图

默认输出位于本文件夹的 `results`，每个问题/τ 包含：

- 三个 setting 各自的 PNG（300 dpi）和 PDF。
- 三个 setting 横向排列、使用统一图例和纵轴的对比 PNG/PDF。
- 各 setting 的原始采样 CSV：算法、运行编号、环境编号、FE/N、MHV。
- `_summary.csv`：各采样位置的均值、样本标准差和样本数。
- JSON：运行参数和绘图窗口，防止误用其他参数的旧数据。

曲线为独立运行的均值，阴影为 ±1 样本标准差，虚线标记第二阶段开始。单次运行时标准差为零。

曲线采样不会逐点打印 MHV；每次运行结束会打印两阶段的采样数量。阶段末尾的 `Stage 2 Locked` 会按本次两阶段预算补记，即使 DTAEA 的更新跳过末尾的名义采样编号也会输出。`TargetGen` 是目标末尾编号，`TriggerGen` 是检测编号；每轮整批评估可能使实际 FE 略超过预算，曲线只保存本次窗口内的数据。

已有匹配参数的 CSV 会直接读取，无需重新实验。重画时沿用生成 CSV 时的参数，并加 `--plot-only`，例如：

```powershell
python run_experiments.py --problems DTLZ1 --taus 50 --settings 1,2,3 --runs 5 --plot-only
```

使用 `--no-std-band` 隐藏阴影，`--sample-every 5` 降低采样频率，`--output-dir` 指定输出目录，`--overwrite` 重新运行并替换相同名称的结果。

原实验 MAT 只含环境末尾的 MHV，无法恢复连续曲线；本程序重新运行算法进行采样。

输出文件名以 `MHV_Env1_Env2` 开头。`smoke_results` 和 `smoke_comparison` 内保留的 `MHV_Env2_Env3` 文件为此前窗口的验证结果，新窗口的验证结果位于 `smoke_env1_env2`。
