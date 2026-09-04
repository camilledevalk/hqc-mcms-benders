# Quantum Resource Estimation

In the project, we also did a loose quantum resource estimation. Loose here means that we made many assumptions, but we still try to come to a reasonable estimation. In this file, we will go through our methods, list our assumptions and approximations, and give a resource estimate for running a [Hybrid Quantum-Classical Multiple-Cut Multiple-Solution Benders decomposition](../MILP-Engine/Overview.md) [1] on a gate-based quantum computer using QAOA and the [minimum set cover strategy](../MILP-Engine/Strategies.md) for cut selection.

## Methods

### Scenarios

Exact resource estimation of the quantum approximate optimization algorithm (QAOA) is generally hard too do, if not impossible, because you don't know in advance how many times you have to run the circuit. This depends on the meta-optimization loop that finds the optimal parameters $\vec{\beta}$ and $\vec{\gamma}$. Also, the QAOA depth $p$ is generally hard to estimate in advance.

To still do meaningful resource estimation, we used different scenario's to do the resource estimation. These scenario's are:

  - **Scenario 1** "quantum optimisation works better than expected" yields 2 QAOA reps and 200 optimisation steps. Quantum computers have a high speed (CLOPS) and low error (EPLG)
  - **Scenario 2** "quantum optimisation works as we currently estimate" yields 10 QAOA reps and 500 optimisation steps. Quantum computers have a reasonable speed (CLOPS) and reasonable error (EPLG)
  - **Scenario 3** "quantum optimisation works worse than expected" yields 30 QAOA reps and 1500 optimisation steps. Quantum computers are not much faster than today and not much more noise-resistant than today (EPLG).
  - Note: the exact numbers for the CLOPS and EPLG will be elaborated on later.

### Approximating number of gates

To get to the number of gates for the cut selection algorithm, we need to find a relation between the number of cut selection variables $n_{\text{cut selection}}$ and $d$ and $n_{\text{qubits}}$ the quantum gate depth and number of qubits respecitvely. In [labjournal 7](../../notebooks/research/labjournal/10.%20estimate_quantum_runtime_revisited.ipynb), we calculate the relation between these quantities. In theory, it is possible to get an exact relation, but we had to calculate and extrapolate.

We found there's a quadratic relationship between the number of cut selection variables and the depth of the (transpiled) circuit, i.e.,

$$
d \propto n_{\text{cut selection}}^2.
$$

See also this plot, which is made with a random binary indicator matrix (erdos renyi random graph with p=0.5):

![relation-depth-cut-selection](../../results/20250828/153352-cut_selection_vars_vs_transpiled_circuit_depth/figures/153353-689287/cut_selection_vars_vs_transpiled_circuit_depth.png)

#### Full approximation flow

The workflow for approximating the gate depth (and number of gates) for a given binary indicator matrix now is as follows:


- Use a strategy to create the cut selection problem (baseline: [`MinimumSetCoverStrategy()`](../../milp_engine/strategies/minimum_set_cover_strategy.py))
- Use [`lp_to_qubo`](../../milp_engine/utils/quantum_utils.py) to convert the linear program to a QUBO.
  - Empirically (lab journal 8.), we've found that there's a relation between the number of cut selection variables and the number of QUBO variables following $n_{\text{QUBO}} \approx n_{\text{cut selection}} \log_2{n_{\text{cut selection}}}$.
- Use the `.to_ising()` method of the QUBO object to convert the QUBO to an Ising operator.
- Then, this one operator can be made into a QAOA cost operator using the `PauliEvolutionGate`.
- The depth of the QAOA is the depth of the cost operator + mixer operator ($d_{\text{mixer}} = n_{\text{QUBO}}$) times the QAOA repetitions $p$.
- Then, it can be transpiled for hardware (or not), depending on the choice for hardware with all-to-all connectivity (like trapped ions) or hardware with limited connectivity (like superconducting qubits).

**Note**: We only do the exact calculation of the number of gates in the QAOA circuit if the number of cut selection variables is less than 50, otherwise, we do the exact for random binary indicator matrices with n=[1, 5, 10, 25, 50] and fit a quadratic scaling.


### Quantum Runtime from an arbitrary circuit

We follow the method developed by IBM to estimate quantum runtime [1]. The very short summary is that error mitigated runtime $J$ for a quantum circuit with $n$ qubits of depth $d$ can be approximated to be

$$
\begin{equation}
    J = \bar{\gamma}^{n\cdot d} \beta d
\end{equation}
$$

with $\bar{\gamma}$ a performance metric (noise) and $\beta$ a speed metric.

Specifically, we can relate $\bar{\gamma}$ and $\beta$ to reported performance metrics as

$$\bar{\gamma} = \frac{1}{(1 - \text{EPLG})^2}$$

with EPLG the error per layered gate and

$$\beta = \frac{1}{\text{CLOPS}_h}$$

with CLOPS$_h$ the [updated](https://www.ibm.com/quantum/blog/quantum-metric-layer-fidelity#introducing-an-updated-clops) circuit layer operations per second metric introduced by IBM.

### Error mitigation vs. error correction

Importantly, this calculation is based on a quantum computer performing _error mitigation_ and _not_ error correction. With error mitigation, you run the circuit multiple times and with statistical and machine learning methods, you try to mitigate errors with a classical computer. This is different from quantum error _correction_, where errors are corrected in the quantum calculation.

Error mitigation does not scale well, meaning that the classical and quantum resources needed grow fast (in fact, exponentially) as the number of qubits or the depth of the circuits you're running increases (see [approximating number of gates](#approximating-number-of-gates) ).

### Full list of assumptions and approximations

- QAOA depth is assumed.
- The number of optimization steps is assumed.
- The relation between $n_{\text{QUBO}}$ and $n_{\text{cut selection}}$ is empirically estimated. This could be verified exactly.
- The quantum circuit depth is extrapolated from a few instances. In theory, there's an exact mapping possible.

## Results

Results will be generated from [this notebook](../../notebooks/research/quantum_resource_estimation.ipynb).

In other words: this wiki page is work in progress.

### Current quantum hardware

### Future (2029) quantum hardware

## Discussion

### QAOA

### Quantum Annealing

### Digitized Counterdiabatic Quantum Optimization (Kipu Quantum's DCQO)

## Future outlook

- Azure Quantum Resource Estimator for quantum error correction
- Different algorithms (e.g., Q-CTRL's or Kipu Quantum's optimisation algorithms)

## References

[1] D. C. McKay, I. Hincks, E. J. Pritchett, M. Carroll, L. C. G. Govia, and S. T. Merkel, ‘Benchmarking Quantum Processor Performance at Scale’, Nov. 10, 2023, *arXiv*: arXiv:2311.05933. doi: 10.48550/arXiv.2311.05933.
