# Pure Python MHV Experiment

This folder is the start of a MATLAB-free port of `RunExperiments_WithMHV.m`.

Current status:

- Ported: `D_MIXED_SEQ` for `DTLZ1-4` and `WFG4-9`.
- Ported: `MHV_Strict` style MHV calculation.
- Ported: experiment grid and result saving.
- Ported: `DTAEA`, `KTDMOEA`, `LEC`, `STA`, `ArchiveSPEA2SDE`,
  `TrustArchiveSPEA2SDE`, `ArchiveNSGAII`,
  `TrustArchiveNSGAII`, and `RestartNSGAII`.
- Included only for pipeline testing: `RandomSearch`.

The four algorithms follow their original-paper logic for reconstruction,
knowledge transfer, mating, and population update. The experiment-specific
stage-signature detection and final-generation MHV recording are retained.
The supplied compact experiment uses 100 generations before the first change,
then four changes across five environments. Some original papers use 1000
generations before the first change.

Run a smoke test:

```powershell
cd "D:\MATLAB\PlatEMO-master\PlatEMO\Problems\Multi-objective optimization\DTLZ\python_mhv_experiment"
python run_experiments.py --runs 1 --taus 25 --settings 1 --algorithms RandomSearch
```

Run the nine default comparison algorithms:

```powershell
python run_experiments.py --runs 1 --taus 25 --settings 1 --problems DTLZ1,DTLZ2,DTLZ3,DTLZ4,WFG4,WFG5,WFG6,WFG7,WFG8,WFG9 --algorithms DTAEA,KTDMOEA,LEC,STA,ArchiveSPEA2SDE,TrustArchiveSPEA2SDE,ArchiveNSGAII,TrustArchiveNSGAII,RestartNSGAII
```

On a cloud server:

```bash
pip install numpy scipy
python run_experiments.py --runs 1 --taus 25 --settings 1 --algorithms RandomSearch
```

Compact formal experiment grid:

```bash
python run_experiments.py --runs 31 --n 150 --taus 25,50,100 --settings 1,2,3 --algorithms DTAEA,KTDMOEA,LEC,STA,ArchiveSPEA2SDE,TrustArchiveSPEA2SDE,ArchiveNSGAII,TrustArchiveNSGAII,RestartNSGAII --pf-count 5000
```

`--problems` defaults to all ten experiment problems. Use a subset such as
`--problems DTLZ1,WFG4` for targeted tests.

The default MHV settings match `MHV_Strict.m`: PF count `5000`, population
and PF Monte Carlo samples `1000000`, exact 2D HV,
and exact 3D HV when the nondominated point count is below `500`.

For the supported minus-DTLZ/minus-WFG benchmarks, the PF-HV denominator is
computed analytically from the normalized true Pareto front, as required by
the benchmark MHV definition. The true ideal and nadir points are also
computed analytically, so finite `GetOptimum()` sampling does not bias the
normalization or the maximum-achievable PF-HV in high dimensions.

`--pf-count` is retained for fallback compatibility with unsupported problem
types. It does not change the analytic PF-HV or true bounds of the ten
supported experiment problems.

Plot MHV recovery curves against FE/N for selected algorithms and problems:

```bash
python plot_mhv_fe.py --algorithms ArchiveSPEA2SDE,ArchiveNSGAII --problems DTLZ1 --setting 1 --tau 50 --runs 5 --sample-every 10
```

The script runs new trials and writes a PNG figure plus its sampled trace CSV
to `MHV_FE_Plots/`. Dashed vertical lines mark environment changes; with two
or more runs, the translucent band is one standard deviation around the mean.

```bash
python run_experiments.py --runs 1 --taus 25 --settings 1 --algorithms RandomSearch --pf-count 1000
```

Note: this is a NumPy implementation of the algorithms and PlatEMO utilities,
not a MATLAB Engine wrapper. NumPy and MATLAB random streams, PCA, LHS, and
nondominated-sort tie behavior are not bitwise identical.

## Independent single-transition benchmark

`run_single_transition_experiments.py` extracts each adjacent change from the
old five-environment sequence into an independent two-environment run.  Every
run starts from a fresh source population, optimizes it, performs exactly one
objective-set change, lets the algorithm construct its new-environment
candidates, and then optimizes the target Pareto front.  No earlier transition
can influence a later one.

The file can be run directly from PyCharm.  Its defaults reproduce the paper's
RAAT-NSGAII configuration for DTLZ2 transition 1 with `N=150`, 100 source
generations, 50 target generations, and 31 independent runs.  A targeted command is:

```powershell
python run_single_transition_experiments.py --runs 31 --problems DTLZ2 --settings 1 --transitions 1 --algorithms RAATNSGAII --n 150 --source-gens 100 --target-gens 50
```

To split the complete old grid into independent transitions:

```powershell
python run_single_transition_experiments.py --runs 31 --problems DTLZ1,DTLZ2,DTLZ3,DTLZ4,WFG4,WFG5,WFG6,WFG7,WFG8,WFG9 --settings 1,2,3 --transitions 1,2,3,4 --algorithms DTAEA,KTDMOEA,LEC,STA,ArchiveSPEA2SDE,TrustArchiveSPEA2SDE,ArchiveNSGAII,TrustArchiveNSGAII,RestartNSGAII --source-gens 100 --target-gens 50
```

Results are written to `Results_SingleTransition/`.  Each `.mat` file contains
exactly two MHV rows per run: the source front immediately before the change
and the target front at the end of the target-environment budget.
