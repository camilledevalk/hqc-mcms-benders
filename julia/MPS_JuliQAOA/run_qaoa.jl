"""
    run_qaoa.jl

Run MPS-based QAOA using JuliQAOA from a JSON input file.

Usage:
    julia --project=julia/MPS_JuliQAOA julia/MPS_JuliQAOA/run_qaoa.jl input.json output.json

Input JSON schema:
{
    "z_interactions": [
        {"qubits": [1, 2], "weight": 1.0},
        {"qubits": [2, 3], "weight": -0.5},
        {"qubits": [1],    "weight": 0.3}
    ],
    "nqubits": 3,
    "p": 1,
    "angles": [0.5, 0.3],          # [beta_1,...,beta_p, gamma_1,...,gamma_p], optional
    "optimize": true,               # whether to optimize angles (default: true)
    "optimizer": "cobyla",         # "cobyla", "nelder_mead" or "particle_swarm" (default: "cobyla")
    "maxiter": 1000,                # max optimizer iterations (default: 1000)
    "cutoff": 1e-6,                 # MPS truncation cutoff (default: 1e-6)
    "maxdim": 64,                   # MPS max bond dimension (default: 64)
    "n_shots": 1000,                # number of bitstring samples (default: 1000)
    "rhobeg": 0.1                   # COBYLA initial step size (optional)
}

Angle convention: [beta_1,...,beta_p, gamma_1,...,gamma_p]
  - beta_i  = mixer angles
  - gamma_i = cost/phase angles

Output JSON schema:
{
    "expectation_value": -1.234,
    "angles": [0.1, 0.2],
    "p": 1,
    "nqubits": 3,
    "converged": true,
    "iterations": 42,
    "status": "success",
    "samples": {"010": 523, "101": 477}
}
"""

using LinearAlgebra
using JuliQAOA
using JSON
using Optim
import NLopt
using ITensorMPS: MPS, orthogonalize!, sample
using ITensors: apply, inner, normalize!



"""
    load_input(path::AbstractString) -> AbstractDict{String,Any}

Parse `path` as JSON and return the top-level object.

This is used by [`run`] to read the input JSON written by
`milp_engine.solvers.quantum_solver.JuliaMPSBackend`.
"""
function load_input(path::String)::AbstractDict{String,Any}
    return JSON.parsefile(path)
end

"""
    build_interactions(input::AbstractDict{String,Any}) -> Vector{ZInteractions}

Convert the input JSON dict into JuliQAOA `ZInteractions` terms.

Expects `input["z_interactions"]` to be a list of dict-like objects of the form
`{"qubits": [i, j, ...], "weight": w}`.

Notes:
- Qubit indices are **1-based** (Julia convention).
- `weight` defaults to `1.0` if omitted.
"""
function build_interactions(input::AbstractDict{String,Any})::Vector{ZInteractions}
    interactions = ZInteractions[]
    for term in input["z_interactions"]
        qubits = convert(Vector{Int}, term["qubits"])
        weight = Float64(get(term, "weight", 1.0))
        push!(interactions, ZInteractions(qubits, weight))
    end
    return interactions
end

"""
    build_final_mps(problem, angles; cutoff, maxdim)

Apply the QAOA variational circuit and return the final MPS state.
"""
function build_final_mps(
    problem::QAOAProblem,
    angles::Vector{Float64};
    cutoff::Float64 = 1e-6,
    maxdim::Int = 64,
)::MPS
    U = JuliQAOA.variational_circuit(problem, angles)
    return apply(U, problem.psi0; cutoff = cutoff, maxdim = maxdim)
end

"""
    sample_bitstrings(ψ::MPS, n_shots::Int)

Sample `n_shots` bitstrings from the MPS state and return a frequency dictionary.
ITensorMPS.sample returns a Vector{Int} with values 1 or 2 for "Qubit" site type
(1 = |0⟩, 2 = |1⟩), so we subtract 1 to get 0/1 bitstrings.
"""
function sample_bitstrings(ψ::MPS, n_shots::Int)::Dict{String,Int}
    # Orthogonalize to site 1 (required by ITensorMPS.sample)
    ψ_orth = orthogonalize!(copy(ψ), 1)

    # ITensorMPS.sample requires the MPS to be normalized and will error if
    # the norm deviates even slightly due to truncation.
    normalize!(ψ_orth)

    counts = Dict{String,Int}()
    for _ = 1:n_shots
        raw = sample(ψ_orth)
        # Convert: 1 → '0', 2 → '1'
        bitstring = join(string(b - 1) for b in raw)
        counts[bitstring] = get(counts, bitstring, 0) + 1
    end
    return counts
end

"""
    run(input_path::AbstractString, output_path::AbstractString) -> Nothing

Run MPS-based QAOA for the problem specified by `input_path` and write results
as JSON to `output_path`.

This function implements the contract used by the Python backend
`milp_engine.solvers.quantum_solver.JuliaMPSBackend`, which calls this script
via `subprocess` with `--project=julia/MPS_JuliQAOA`.

Required input keys:
- `"z_interactions"`: list of `{ "qubits": [Int...], "weight": Float }` terms.
- `"p"`: QAOA depth.

Optional input keys:
- `"nqubits"`: defaults to the maximum qubit index in `z_interactions`.
- `"angles"`: initial (or fixed) angles in the convention
  `[beta_1, ..., beta_p, gamma_1, ..., gamma_p]`.
- `"optimize"` (default: `true`): whether to optimize angles.
- `"optimizer"` (default: `"cobyla"`): `"cobyla"`, `"nelder_mead"`, or `"particle_swarm"`.
- `"maxiter"` (default: `1000`): optimizer iterations.
- `"cutoff"` (default: `1e-6`): MPS truncation cutoff.
- `"maxdim"` (default: `64`): maximum MPS bond dimension.
- `"n_shots"` (default: `1000`): number of bitstring samples to draw.
- `"rhobeg"` (optional): COBYLA initial step size (NLopt `initial_step`).

Output keys include `"expectation_value"`, `"angles"`, `"p"`, `"nqubits"`,
`"status"`, and optionally `"samples"`.
"""
function run(input_path::String, output_path::String)::Nothing
    input = load_input(input_path)

    # Build problem
    interactions = build_interactions(input)
    nqubits = get(input, "nqubits", maximum(maximum(i.qubits) for i in interactions))
    problem = QAOAProblem(interactions; nqubits = nqubits)

    p = input["p"]
    cutoff = get(input, "cutoff", 1e-6)
    maxdim = get(input, "maxdim", 64)
    do_optimize = get(input, "optimize", true)
    n_shots = get(input, "n_shots", 1000)

    result = Dict{String,Any}()
    best_angles = Float64[]

    if do_optimize
        optimizer_name = lowercase(String(get(input, "optimizer", "cobyla")))
        maxiter = get(input, "maxiter", 1000)

        # Initial angles: provided or random in [0, pi]
        init_angles = if haskey(input, "angles")
            convert(Vector{Float64}, input["angles"])
        else
            rand(2 * p) .* pi
        end

        # Objective: minimize expectation value via MPS
        objective(angles) = run_qaoa_mps(angles, problem; cutoff = cutoff, maxdim = maxdim)

        # Accumulate per-evaluation trace (used by all optimizers)
        eval_trace = Dict{String,Any}[]
        eval_count = Ref(0)
        function traced_objective(angles)
            val = objective(angles)
            eval_count[] += 1
            push!(eval_trace, Dict{String,Any}(
                "iteration" => eval_count[],
                "value" => val,
                "angles" => collect(Float64, angles),
            ))
            return val
        end

        opt_options = Optim.Options(
            iterations = maxiter,
            show_trace = true,
            show_every = 50,
            store_trace = true,
            extended_trace = true,
        )

        if optimizer_name == "nelder_mead"
            res = optimize(
                traced_objective,
                init_angles,
                NelderMead(),
                opt_options,
            )

            best_angles = collect(Optim.minimizer(res))
            result["angles"] = best_angles
            result["expectation_value"] = Optim.minimum(res)
            result["converged"] = Optim.converged(res)
            result["iterations"] = Optim.iterations(res)
        elseif optimizer_name == "particle_swarm"
            lower = fill(-2pi, 2 * p)
            upper = fill(2pi, 2 * p)
            res = optimize(
                traced_objective,
                lower,
                upper,
                init_angles,
                ParticleSwarm(),
                opt_options,
            )

            best_angles = collect(Optim.minimizer(res))
            result["angles"] = best_angles
            result["expectation_value"] = Optim.minimum(res)
            result["converged"] = Optim.converged(res)
            result["iterations"] = Optim.iterations(res)
        elseif optimizer_name == "cobyla"
            # COBYLA implementation via NLopt (derivative-free).
            lower = fill(-2pi, 2 * p)
            upper = fill(2pi, 2 * p)

            opt = NLopt.Opt(:LN_COBYLA, length(init_angles))
            opt.lower_bounds = lower
            opt.upper_bounds = upper
            opt.maxeval = maxiter
            opt.ftol_rel = 1e-6

            rhobeg = get(input, "rhobeg", nothing)
            if rhobeg !== nothing
                NLopt.initial_step!(opt, rhobeg)
            end
            NLopt.min_objective!(opt, (x, _grad) -> traced_objective(x))

            (minf, minx, ret) = NLopt.optimize(opt, init_angles)
            best_angles = collect(minx)

            result["angles"] = best_angles
            result["expectation_value"] = minf
            result["converged"] = ret in (:FTOL_REACHED, :XTOL_REACHED, :STOPVAL_REACHED)
            result["iterations"] = NLopt.numevals(opt)
        else
            error(
                "Unknown optimizer: $optimizer_name. Use \"cobyla\", \"nelder_mead\", or \"particle_swarm\".",
            )
        end

        result["optimization_trace"] = eval_trace
    else
        # Just evaluate at the given angles (no optimization)
        best_angles = convert(Vector{Float64}, input["angles"])
        if length(best_angles) != 2 * p
            error("Expected $(2*p) angles (2*p), got $(length(best_angles)).")
        end
        ev = run_qaoa_mps(best_angles, problem; cutoff = cutoff, maxdim = maxdim)
        result["angles"] = best_angles
        result["expectation_value"] = ev
    end

    # Sample bitstrings from the final MPS state
    if n_shots > 0
        ψ_f = build_final_mps(problem, best_angles; cutoff = cutoff, maxdim = maxdim)
        result["samples"] = sample_bitstrings(ψ_f, n_shots)
    end

    result["p"] = p
    result["nqubits"] = nqubits
    result["status"] = "success"

    open(output_path, "w") do f
        JSON.print(f, result, 2)
    end

    println("Result written to $output_path")
    println("  Expectation value: $(result["expectation_value"])")
    println("  Angles: $(result["angles"])")
    if haskey(result, "samples")
        println(
            "  Samples: $(length(result["samples"])) unique bitstrings from $n_shots shots",
        )
    end
end

"""
    main(args::Vector{String}=ARGS) -> Nothing

CLI entry point for this script.

Usage:
    julia --project=julia/MPS_JuliQAOA julia/MPS_JuliQAOA/run_qaoa.jl input.json output.json

The expected input and output JSON schemas are described in the docstring of
[`run`].
"""
function main(args::Vector{String} = ARGS)::Nothing
    if length(args) != 2
        println(stderr, "Usage: julia run_qaoa.jl <input.json> <output.json>")
        exit(1)
    end
    run(args[1], args[2])
end

# --- Entry point ---
if length(ARGS) != 2
    println(stderr, "Usage: julia run_qaoa.jl <input.json> <output.json>")
    exit(1)
end

run(ARGS[1], ARGS[2])
