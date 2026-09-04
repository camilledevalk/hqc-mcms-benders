# Project Overview

This project addresses a logistics challenge commonly encountered in military operations: supplying frontline units with essential resources such as food, ammunition, and equipment. The task involves routing a limited number of vehicles—potentially of different modalities, e.g. trucks, drones, helicopters—across various units. The objective is to determine routes that are fast, safe (minimizing expected loss of equipment and human lives), and efficient in terms of vehicle usage and capacity.

This problem can be seen as an instance of the [Vehicle Routing Problem (VRP)](VRP/Vehicle-Routing-Problem-Formulations.md), a well-known "hard" problem in computer science. Large instances—typically involving dozens to hundreds of vehicles and destinations—cannot be solved optimally, not manually nor with classical computing methods. The runtime of algorithms that guarantee *optimal* solutions grows exponentially with the size of the problem. And while finding *good* (but not necessarily optimal) solutions is more tractable, it remains computationally demanding. Smart algorithms for VRP that manage to find *good* solutions quick could therefore have a significant impact on real-world military logistics.

[Quantum computers](https://en.wikipedia.org/wiki/Quantum_computing) offer promising scaling properties for certain types of problems, such as factoring integers, simulating quantum systems, and solving unstructured search problems. However, current quantum hardware is limited: devices typically support only tens to low hundreds of qubits, and are plagued by noise that restricts circuit depth and complexity. Moreover, VRP is [NP-hard](https://en.wikipedia.org/wiki/NP-hardness), and quantum computers are not expected to overcome the exponential runtime barrier for such problems. Nonetheless, there may be ways to leverage **hybrid quantum-classical** computing to accelerate the search for *good* solutions to VRPs. This is the focus of our project, Logistiqs.

## Method

Our approach is inspired by the hybrid quantum-classical method described in the paper *Hybrid Quantum-Classical Multi-cut Benders Approach with a Power System Application* by Nikolaos G. Paterakis (2023) [1]. It combines classical and quantum computing to solve a [Mixed Integer Linear Program (MILP)](https://en.wikipedia.org/wiki/Integer_programming) formulation of the VRP.

We use the [Miller–Tucker–Zemlin (MTZ) formulation](https://en.wikipedia.org/wiki/Travelling_salesman_problem#Integer_linear_programming_formulations) of the VRP, which expresses the problem as a MILP involving both binary (integer) and continuous variables. The presence of integer variables makes MILPs hard to solve. To address this, we apply [Benders Decomposition (BD)](https://en.wikipedia.org/wiki/Benders_decomposition), an iterative method that separates the problem into subproblems that contain only the continuous part of the problem (easily solvable). BD solves these subproblems to generate cuts that constrain the solution space of the integer variables. Over successive iterations, the algorithm tightens upper and lower bounds on the objective value until convergence is achieved.

BD offers several advantages for our use case. First, a general-purpose MILP solver is more versatile than a solver tailored to a specific VRP variant. This flexibility is valuable in military contexts, where logistical requirements can shift rapidly as conflicts evolve. Second, BD allows for early termination: even if the optimal solution has not yet been found, the algorithm provides a feasible solution along with a bound on how far it may be from optimal. This is useful in scenarios where finding good solutions quickly is more important than guaranteeing optimality.

Despite its benefits, BD is known for slow convergence. Paterakis addresses this by solving multiple subproblems per iteration, generating multiple cuts, and selecting an optimal combination of cuts using heuristics. These heuristics aim to choose cuts that maximally constrain the integer variables. This is one of many known ways in which BD can be accelerated [2].
However, the cut-selection process itself is a hard combinatorial problem. Paterakis proposes solving this via [quantum annealing](https://en.wikipedia.org/wiki/Quantum_annealing), converting the cut-selection problem into a Quadratic [Unconstrained Binary Optimization (QUBO)](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization) problem and solving it using a D-Wave quantum annealer.

While promising, D-Wave’s approach comes with notable drawbacks. The hardware suffers from poor qubit connectivity, requiring many qubits to be sacrificed for embedding, and the process of mapping problem connectivity to hardware is computationally expensive—often dominating the runtime. In addition, the service operates as a black box, making it difficult to discern which parts of the computation are genuinely quantum. This complicates research and reproducibility. Finally, D-Wave has no clear path to scaling beyond 5000 qubits, raising concerns about long-term viability.

We opt for [gate-based](https://en.wikipedia.org/wiki/Quantum_logic_gate) quantum computers rather than annealers. On gate-based hardware, a suitable algorithm for solving QUBOs is the [Quantum Approximate Optimization Algorithm (QAOA)](https://arxiv.org/abs/1411.4028) [3][4], which we adopt for this project.

In summary, we formulate the VRP as a MILP and solve it using Benders Decomposition. We accelerate BD convergence using heuristics, and we use quantum computing—specifically QAOA on gate-based devices—to optimize the heuristic cut-selection step. The goal is to find good solutions to large-scale VRPs relevant to military logistics within reasonable timeframes.

## Scope

The project involves implementing the Benders Decomposition method as described by Paterakis [1], with readable, maintainable, and well-documented code suitable for handover to the Ministry of Defense. We define a simplified VRP that includes multiple vehicle modalities and stochastic survival probabilities, but excludes features such as time constraints, dynamic environments, and redundancy planning.

The project also has a research component. This includes empirical evaluation of the following:

- Comparison with non-quantum state-of-the-art methods for our VRP variant.
- Assessment of how heuristics from Paterakis improve BD convergence.
- Analysis of runtime contributions from the cut-selection step in BD. If this step is computationally intensive, quantum computing may offer significant advantages—provided it can solve the associated QUBOs efficiently.
- Evaluation of QAOA performance on the QUBOs generated by our algorithm, including solution quality and scalability as the number of subproblems increases.
- Analytical predictions of performance on future quantum hardware, based on empirical runtimes and IBM’s 2029 hardware roadmap.

## Limitations

The [VRP model used in this project](VRP/Stochastic-Multi-Modal-Multi-Depot-VRP.md) does not aim to be fully realistic. Real-world military logistics involves complexities such as day-night cycles, weather conditions, enemy strategies, redundancy requirements (as one major noted: “one route is no route”), temporal planning, and dynamic adaptation to changing conditions. Our goal is not to model these in full, but rather to create a representative abstraction and explore whether quantum computing can offer advantages in solving such problems.

The practicality of solving MILPs with Bender's Decomposition in this context all depends on how fast the algorithm starts producing good, feasible solutions. Convergence of BD can be accelerated in many different ways [2] and, in this project, we focus exclusively on one: heuristic cut selection. Hence, even if is possible to accelerate BD with our method, it remains uncertain how much benefit would persist when combined with other acceleration techniques.

## References
- [1] N. G. Paterakis, ‘Hybrid quantum-classical multi-cut Benders approach with a power system application’, Computers and Chemical Engineering, 2023.
- [2] R. Rahmaniani al., ‘The Benders Decomposition Algorithm: A Literature Review’, European Journal of Operational Research, 2016.
- [3] E. Farhi et al., ‘A Quantum Approximate Optimization Algorithm’, arXiv, 2014.
- [4] A. Abbas et al., ‘Challenges and Opportunities in Quantum Optimization’, Nature Reviews Physics, 2024.
- [5] C. de Valk, K. Reerink, S. Sebus, and S. de Bon, ‘Hybrid quantum-classical end-to-end pipeline for solving MILPs: a vehicle routing case study’, arXiv:2607.26771 [quant-ph], 2026. [https://doi.org/10.48550/arXiv.2607.26771](https://doi.org/10.48550/arXiv.2607.26771)
