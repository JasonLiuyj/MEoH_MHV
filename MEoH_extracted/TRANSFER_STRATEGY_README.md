# 用 MEoH 搜索动态多目标 Transfer Strategy

## 问题背景

在动态多目标优化中，环境变化会改变有效目标集合和目标数量：目标可能增加、减少或被替换。例如，环境1使用目标 `[2, 4]`，环境2既可能扩展为 `[2, 4, 5]`，也可能收缩为 `[2]`。旧 archive 的目标值属于旧目标空间，不能直接当作新环境中的完整目标信息。

环境1中的 SPEA2+SDE 已经产生一个 archive，其中保存决策向量和旧目标值。环境变化时，新目标尚未评价，因此需要从大量旧解中选出值得迁移的决策向量，在环境2重新评价，并用它们构造环境2初始种群。

本实验固定以下部分：

- 基础优化器为 SPEA2+SDE；
- 动态问题为 DTLZ2；
- 环境变化、种群规模、迁移比例和评价预算；
- 新环境中的重新评价、随机补齐和 MHV 计算。

MEoH 只搜索函数：

```python
score_archive_solutions(archive_decs, archive_objs, old_active, new_active)
```

函数为每个 archive 非支配解返回一个有限优先级，分数越大越优先迁移。种群规模与正式实验一致，固定为 `N=150`。若旧环境第一非支配前沿不超过150个，则全部迁移并随机补足到150；若超过150个，则按策略得分选取前150个。这与 `ArchiveSPEA2SDE` 的迁移数量和补齐机制一致，不再使用固定50%迁移比例。

两个原始评价指标分别是：

1. 从环境变化后新种群开始，直到环境2结束的 MHV 曲线归一化 AUC，越大越好。曲线每5代采样一次并强制包含起点和终点；
2. transfer strategy 对 archive 打分、排序并选出迁移解的时间，越小越好。

由于 MEoH 内部统一执行最大化，第二项传给 MEoH 的实际分数是负耗时 `-selection_time`。计时取5次重复的中位数，并且只包含 transfer strategy 的评分、校验、排序和选择，不包含新环境目标函数重评价及后续 SPEA2+SDE 进化。

评价覆盖 mild、moderate、severe 三个确定性目标扩张场景。当前入口严格使用原始 MEoH 的 E2、M1、M2，关闭 P1、T1、TIDE、RAISE、PFG、聚类、反思和 Harmony Search。

## 运行

确认 `.env` 中已经配置 LLM API，然后在 PyCharm 中右键运行：

```text
run_transfer_strategy_meoh.py
```

结果保存到：

```text
logs/transfer_strategy_original_meoh
```

