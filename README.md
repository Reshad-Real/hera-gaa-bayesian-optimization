# HERA: constrained multi-objective Bayesian optimization of GAA transistors with TCAD in the loop and a calibrated digital twin

Code and data for the manuscript:

> Md. Reshad Al Muttaki, Tahmid Faiz Noor, SK Tahmed Salim Rafid. *Constrained multi-objective Bayesian optimization of gate-all-around transistors with TCAD in the loop and a calibrated digital twin.* Submitted to *Computers & Electrical Engineering*.

HERA is a constrained multi-objective Bayesian optimization framework for expensive engineering design. It combines GP-ensemble surrogates, a constrained expected-hypervolume-improvement acquisition maximized by an NSGA-II search, greedy batch selection and a PPO warm-start. Optionally, it uses a calibrated model or earlier simulations as prior knowledge. The repository contains the optimizer, the baselines, the DEVSIM drift-diffusion TCAD reference, the TCAD-calibrated digital twin of a 3/2 nm-class gate-all-around transistor, and the raw results behind every table and figure of the paper.

## Code (`hera_twin/`)

| File | Purpose |
|---|---|
| `hera.py` | HERA optimizer: GP ensemble, constrained EHVI, NSGA-II acquisition optimizer, PPO warm-start, greedy batch, cost-aware gate |
| `baselines.py` | Baselines: random search (Sobol), NSGA-II, ParEGO, pool-based q-EHVI, q-EHVI with a multi-start L-BFGS-B acquisition optimizer |
| `gp_surrogate.py` | GP-ensemble surrogates, with optional prior knowledge (Section 3.8): the twin as prior mean (`PRIOR`) and the calibration simulations as extra training data (`EXTRA`) |
| `devsim_sim.py`, `devsim_campaign.py`, `tcad_merge.py` | DEVSIM 2-D drift-diffusion reference (doping/field-dependent mobility, Gaussian S/D, gate charge) and parallel TCAD campaigns |
| `finfet_oracle.py`, `finfet_geometry.py`, `calibrate_twin.py` | TCAD-calibrated compact digital twin (Appendix A) and its calibration |
| `config.py` | Design box, constraints (IRDS-based), hypervolume reference frame, hyperparameters |
| `study.py`, `extra_study.py` | 30-seed device study |
| `ablation_study.py`, `ablation_extra.py` | Component ablation (15 seeds) |
| `benchmark_study.py`, `benchmark_extra.py`, `benchmarks.py` | DTLZ2, DTLZ4 and C2-DTLZ2 benchmarks (12 seeds) |
| `tcad_loop.py`, `tcad_loop_report.py` | Optimization with DEVSIM TCAD in the loop (10 seeds), Table 13 and Fig. 13 |
| `caee_report.py`, `tcad_rescore_extra.py` | Summaries, TCAD re-scoring of the final Pareto sets |
| `paper_tables.py`, `paper_figures.py` | The paper's tables (Markdown) and print-size figures |
| `hypervolume.py`, `nsga2.py`, `rl_controller.py` | Hypervolume, NSGA-II and PPO building blocks |

In `tcad_loop.py`, the run keys are `HERA`, `qEHVI`, `qEHVI-LBFGS` and `Random` without prior knowledge. A suffix selects the prior knowledge given to the surrogates: `-TI` the twin prior, `-D` the calibration data, `-TID` both.

## Data (`results/`)

| File | Content |
|---|---|
| `devsim_calib.npz`, `devsim_calib.json` | 128 Sobol designs simulated in DEVSIM (96 for calibration, 32 held out) |
| `twin_params.json`, `twin_calibration_metrics.json`, `table_calibration.md` | Fitted twin parameters and held-out error metrics |
| `study_results.pkl`, `study_curves.npz`, `study_summary.json` | 30-seed device study: every evaluated design, objective and constraint value, hypervolume histories, statistics |
| `ablation_curves.npz`, `ablation_summary.json` | Component ablation |
| `benchmark_results.pkl`, `benchmark_curves.npz`, `benchmark_summary.json` | DTLZ benchmarks |
| `pareto_for_tcad.npy`, `pareto_for_tcad_index.pkl`, `devsim_pareto.npz` | Final Pareto designs of the device study and their DEVSIM re-simulation (index: seed, method, design) |
| `tcad_rescore_summary.json`, `tcad_parity_summary.json` | TCAD re-scoring of the fronts and twin-vs-TCAD accuracy on the optimized designs |
| `chosen_design.*`, `devsim_chosen.*` | Selected balanced design and its TCAD verification |
| `tcad_loop_results.pkl`, `tcad_loop_cache.pkl`, `tcad_loop_summary.json`, `tcad_loop_log*.txt` | Optimization with TCAD in the loop: all runs, the cache of every DEVSIM simulation, statistics and logs |
| `hv_frame_sensitivity*.json` | Hypervolume under alternative reference frames |
| `table_*_paper.md`, `table_*.md` | Tables of the paper |
| `figures/` | Figures of the paper (PNG and vector PDF) |

Pickle files contain plain Python dictionaries, lists and NumPy arrays. Load them with `pickle.load(open(path, "rb"))`.

## Requirements

Python ≥ 3.11, NumPy, SciPy, scikit-learn, matplotlib, and DEVSIM (≥ 2.10) with Intel MKL (`pip install devsim mkl`). The paper's runs used Python 3.14.5, NumPy 2.4, SciPy 1.17, scikit-learn 1.8 and DEVSIM 2.10 on an Intel Core i7-13650HX laptop.

## Reproducing the paper

```bash
# 1. TCAD calibration data set (128 Sobol designs, ~2 min on 18 cores)
python -m hera_twin.devsim_campaign calib --n 128 --workers 18
# 2. Calibrate the digital twin (writes results/twin_params.json, table_calibration.md)
python -m hera_twin.calibrate_twin
# 3. 30-seed device study, 15-seed ablation, 12-seed DTLZ benchmarks
python -m hera_twin.study --seeds 30 --budget 64 --workers 18
python -m hera_twin.extra_study --workers 18
python -m hera_twin.ablation_study --seeds 15 --budget 64 --workers 15
python -m hera_twin.benchmark_study --seeds 12 --budget 64 --workers 10
python -m hera_twin.benchmark_extra --workers 18
# 4. Summaries
python -m hera_twin.caee_report calib
python -m hera_twin.caee_report study
python -m hera_twin.caee_report bench
python -m hera_twin.caee_report ablation
python -m hera_twin.caee_report hv_frame
python -m hera_twin.caee_report showcase
# 5. TCAD verification of the selected design and of all final Pareto sets
python -m hera_twin.devsim_campaign points --file results/chosen_design.npy --tag chosen --workers 1
python -m hera_twin.caee_report chosen_tcad
python -m hera_twin.caee_report tcad_collect
python -m hera_twin.devsim_campaign points --file results/pareto_for_tcad.npy --tag pareto --workers 18
python -m hera_twin.caee_report tcad_rescore
python -m hera_twin.tcad_rescore_extra
# 6. Optimization with TCAD in the loop (10 seeds, 64 TCAD evaluations each)
python -m hera_twin.tcad_loop init --seeds 10 --workers 18
python -m hera_twin.tcad_loop run --seeds 10 --budget 64 --workers 17 --methods HERA,HERA-TI,HERA-D,HERA-TID,qEHVI,qEHVI-TI,qEHVI-LBFGS,qEHVI-LBFGS-D,Random
python -m hera_twin.tcad_loop_report
# 7. Tables and figures of the paper
python -m hera_twin.paper_tables
python -m hera_twin.paper_figures
```

The repository holds the merged result of each TCAD campaign (`devsim_pareto.npz`), not the intermediate campaign files.

## License

MIT (see `LICENSE`).
