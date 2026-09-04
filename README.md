# Logistiqs — IEEE paper reproduction package

This branch is a trimmed, self-contained snapshot of the Logistiqs project containing everything needed to reproduce the experiments and figures in *"Hybrid quantum-classical end-to-end pipeline for solving MILPs: a vehicle routing case study"* — nothing else. For the full research codebase (broader experiments, the interactive frontend demo, etc.), see the `research` and `dev` branches instead.

A description of the underlying method can be found [here](wiki/Project-Overview.md); a full documentation index is in the [wiki](wiki/Home.md).

## Reproducing the paper

1. **Install dependencies**: Install [uv](https://docs.astral.sh/uv/) and run `uv sync` to create an exact, pinned environment (`pyproject.toml` + `uv.lock`).

2. **Generate VRP instances** (optional — already committed):
   - Run `notebooks/01_generate_vrp_instances.ipynb` to regenerate toy and QOptLib permuted instances.
   - Outputs: `experimental-data/instances/toy/` and `experimental-data/instances/qoptlib-permuted/`.
   - **Skip this step** if you just want to use the committed instances.

3. **Run experiments** (generates most data for figures):
   - **Notebook 02** (`notebooks/02_qoptlib_mcms_experiment.ipynb`): QOptLib MCMS Benders experiment (classical, no quantum). Runs 40 tasks (10 instances × 4 `n_subproblems` values) with 1.5h time budget per task. Worst-case ~60h sequentially, typical ~10-30h. Outputs to `experimental-data/results/qoptlib-mcms/` (gitignored).
   - **Notebook 03** (`notebooks/03_toy_hqc_mcms_experiment.ipynb`): Toy HQC-MCMS Benders experiment with Julia QAOA. Requires Julia with `julia/MPS_JuliQAOA` instantiated (see `julia/README.md`). Runs 10 tasks (5 instances × 2 QAOA depths) with 4h time budget per task. Worst-case ~40h sequentially, typical ~10-30h. Outputs to `experimental-data/results/toy-hqc-mcms/` (gitignored).
   - Both notebooks enforce time budgets gracefully (solver stops, notebook continues) and checkpoint progress (safe to interrupt and resume).

4. **QAOA demo** (optional):
   - Run `notebooks/04_set_cover_qaoa_demo.ipynb` to see how different QAOA backends (MPS-JuliQAOA, Qiskit, Fermioniq) solve a minimum set cover problem. This reproduces the method behind Figure 4 but not the exact archived numbers (see notebook header).

5. **Generate figures**:
   - Run `notebooks/05_generate_figures.ipynb` to regenerate all paper figures.
   - Uses outputs from notebooks 02-03 (if available) plus archived baseline data in `experimental-data/archived/` (Fermioniq/IBM Quantum results that cannot be regenerated without paid services).
   - Outputs: `notebooks/figures/` (PDF figures matching the paper).

## Folder structure

```
experimental-data/
├── instances/              # VRP instances (committed)
│   ├── toy/                # 5 toy instances (n=5 customers)
│   └── qoptlib-permuted/   # 10 QOptLib permutations (n=20 customers)
├── results/                # Experiment outputs (gitignored, user-generated)
│   ├── qoptlib-mcms/       # Notebook 02 outputs
│   └── toy-hqc-mcms/       # Notebook 03 outputs
└── archived/               # Historical baseline data (committed)
    ├── fermioniq-ibm/      # Fermioniq + IBM Quantum QAOA results
    └── julia-qaoa/         # MPS-JuliQAOA baseline results

notebooks/
├── 01_generate_vrp_instances.ipynb      # Generate toy + QOptLib instances (optional)
├── 02_qoptlib_mcms_experiment.ipynb     # QOptLib MCMS experiment (classical)
├── 03_toy_hqc_mcms_experiment.ipynb     # Toy HQC-MCMS experiment (Julia QAOA)
├── 04_set_cover_qaoa_demo.ipynb         # QAOA backend demonstration (optional)
└── 05_generate_figures.ipynb            # Generate all paper figures

milp_engine/                # Core Benders decomposition solver library
julia/                      # MPS-JuliQAOA tensor network emulator integration
wiki/                       # Documentation (method, VRP formulations, CLI usage)
```

## Quick start (skip to plotting)

If you just want to regenerate figures without running experiments:

```bash
uv sync
jupyter notebook notebooks/05_generate_figures.ipynb
```

The notebook uses committed instances + archived data, so it runs immediately without waiting for experiments.

## Direct CLI usage

To call the Bender's decomposition solver (`BenderMILPSolver`) directly from Python instead of through notebooks, see the [BenderMILPSolver CLI wiki](wiki/MILP-Engine/BenderMILPSolver-CLI.md).

## Contact
|Name|Role|Email|
|----|----|-----|
|Camille de Valk|Project Lead & Developer|[camille.de.valk@capgemini.com](mailto:camille.de.valk@capgemini.com)|
|Koen Reerink|Developer||
|Siert Sebus|Developer||
|Myra Coppens|Developer||
