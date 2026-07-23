from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.dft.bandgap import bandgap
from ase.io import write
from ase.optimize import BFGS
from gpaw import FermiDirac, GPAW, MixerSum, PW

CUTOFF_EV = 600.0
KMESH = (8, 8, 8)
FERMI_WIDTH_EV = 0.05
FMAX_EV_A = 0.05
MAX_STEPS = 100
MAXSTEP_A = 0.08
PERTURBATION_A = 0.01
PERTURBATION_SEED = 20260723


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


def build_ideal_structure(lattice_A: float) -> Atoms:
    symbols = ["Pd", "Pd"] + ["H"] * 8
    scaled = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    scaled.extend(list(itertools.product((0.25, 0.75), repeat=3)))
    atoms = Atoms(symbols=symbols, cell=np.eye(3) * lattice_A, pbc=True)
    atoms.set_scaled_positions(np.asarray(scaled, dtype=float))
    return atoms


def perturb_hydrogen(atoms: Atoms) -> np.ndarray:
    rng = np.random.default_rng(PERTURBATION_SEED)
    direction = rng.normal(size=(8, 3))
    direction -= direction.mean(axis=0, keepdims=True)
    direction /= np.linalg.norm(direction, axis=1).max()
    displacement = np.zeros((len(atoms), 3), dtype=float)
    displacement[2:] = PERTURBATION_A * direction
    atoms.positions += displacement
    atoms.wrap()
    return displacement


def force_metrics(forces: np.ndarray, symbols: list[str]) -> dict:
    forces = np.asarray(forces, dtype=float)
    norms = np.linalg.norm(forces, axis=1)
    grouped: dict[str, list[float]] = {}
    for symbol, norm in zip(symbols, norms):
        grouped.setdefault(symbol, []).append(float(norm))
    return {
        "max_force_eV_per_A": float(norms.max()),
        "force_rms_eV_per_A": float(np.sqrt(np.mean(norms**2))),
        "force_norms_eV_per_A": norms.tolist(),
        "per_species": {
            symbol: {
                "max_force_eV_per_A": float(np.max(values)),
                "rms_force_eV_per_A": float(np.sqrt(np.mean(np.square(values)))),
                "mean_force_eV_per_A": float(np.mean(values)),
            }
            for symbol, values in sorted(grouped.items())
        },
    }


def geometry_metrics(atoms: Atoms) -> dict:
    symbols = atoms.get_chemical_symbols()
    hydrogen = [i for i, symbol in enumerate(symbols) if symbol == "H"]
    palladium = [i for i, symbol in enumerate(symbols) if symbol == "Pd"]
    hh = sorted(
        float(atoms.get_distance(i, j, mic=True))
        for offset, i in enumerate(hydrogen)
        for j in hydrogen[offset + 1 :]
    )
    coordination = []
    for index in palladium:
        distances = sorted(float(atoms.get_distance(index, h, mic=True)) for h in hydrogen)
        coordination.append(
            {
                "atom_index": index,
                "nearest_8_H_A": distances[:8],
                "count_H_within_2p25_A": int(sum(distance < 2.25 for distance in distances)),
            }
        )
    return {
        "minimum_HH_A": float(hh[0]),
        "shortest_12_HH_A": hh[:12],
        "h2_like_pair_count_lt_1A": int(sum(distance < 1.0 for distance in hh)),
        "Pd_H_coordination": coordination,
    }


def displacement_metrics(reference: Atoms, target: Atoms) -> dict:
    delta = target.get_scaled_positions(wrap=True) - reference.get_scaled_positions(wrap=True)
    delta -= np.rint(delta)
    cartesian = delta @ np.asarray(reference.cell.array, dtype=float)
    norms = np.linalg.norm(cartesian, axis=1)
    return {
        "rms_A": float(np.sqrt(np.mean(norms**2))),
        "max_A": float(norms.max()),
        "per_atom_A": norms.tolist(),
    }


def make_calculator(text_output: str) -> GPAW:
    return GPAW(
        mode=PW(CUTOFF_EV),
        xc="PBE",
        kpts={"size": KMESH, "gamma": True},
        occupations=FermiDirac(FERMI_WIDTH_EV),
        spinpol=True,
        mixer=MixerSum(beta=0.05, nmaxold=5, weight=50.0),
        symmetry="off",
        maxiter=400,
        convergence={"energy": 1.0e-5, "density": 2.0e-4, "eigenstates": 2.0e-6},
        txt=text_output,
    )


def electronic_snapshot(calc: GPAW) -> dict:
    try:
        fermi_levels = np.atleast_1d(calc.get_fermi_levels()).astype(float)
    except Exception:
        fermi_levels = np.atleast_1d(calc.get_fermi_level()).astype(float)
    nspins = int(calc.get_number_of_spins())
    ibz_kpoints = np.asarray(calc.get_ibz_k_points(), dtype=float)
    weights = np.asarray(calc.get_k_point_weights(), dtype=float)
    max_occ = 1.0 if nspins == 2 else 2.0
    sigma = 0.05
    fractional = 0
    dos_at_ef = 0.0
    minimum_abs = math.inf
    nearest_spectra = []
    eig_blocks = []
    occ_blocks = []
    for spin in range(nspins):
        spin_eigs = []
        spin_occs = []
        ef = float(fermi_levels[min(spin, len(fermi_levels) - 1)])
        for kpoint in range(len(ibz_kpoints)):
            eigenvalues = np.asarray(calc.get_eigenvalues(kpt=kpoint, spin=spin), dtype=float)
            occupations = np.asarray(calc.get_occupation_numbers(kpt=kpoint, spin=spin), dtype=float)
            spin_eigs.append(eigenvalues)
            spin_occs.append(occupations)
            fractional += int(np.count_nonzero((occupations > 1.0e-4) & (occupations < max_occ - 1.0e-4)))
            minimum_abs = min(minimum_abs, float(np.min(np.abs(eigenvalues - ef))))
            dos_at_ef += float(weights[kpoint]) * float(
                np.exp(-0.5 * ((eigenvalues - ef) / sigma) ** 2).sum()
                / (math.sqrt(2.0 * math.pi) * sigma)
            )
            nearest = np.argsort(np.abs(eigenvalues - ef))[:12]
            nearest_spectra.append(
                {
                    "spin": spin,
                    "ibz_kpoint": kpoint,
                    "weight": float(weights[kpoint]),
                    "indices": nearest.tolist(),
                    "eigenvalues_eV": eigenvalues[nearest].tolist(),
                    "occupations": occupations[nearest].tolist(),
                }
            )
        eig_blocks.append(np.asarray(spin_eigs, dtype=float))
        occ_blocks.append(np.asarray(spin_occs, dtype=float))
    if nspins == 1:
        dos_at_ef *= 2.0
    np.savez_compressed(
        "eigenvalues_occupations.npz",
        eigenvalues=np.asarray(eig_blocks, dtype=float),
        occupations=np.asarray(occ_blocks, dtype=float),
        ibz_kpoints=ibz_kpoints,
        kpoint_weights=weights,
        fermi_levels_eV=fermi_levels,
    )
    try:
        indirect_gap = jsonable(bandgap(calc, direct=False))
        direct_gap = jsonable(bandgap(calc, direct=True))
    except Exception as exc:
        indirect_gap = {"error": type(exc).__name__, "message": str(exc)}
        direct_gap = indirect_gap
    return {
        "fermi_levels_eV": fermi_levels.tolist(),
        "nspins": nspins,
        "ibz_kpoint_count": int(len(ibz_kpoints)),
        "fractionally_occupied_state_count": int(fractional),
        "gaussian_dos_at_ef_states_per_eV_cell_sigma_0p05": float(dos_at_ef),
        "minimum_abs_eigenvalue_minus_ef_eV": float(minimum_abs),
        "indirect_bandgap": indirect_gap,
        "direct_bandgap": direct_gap,
        "near_fermi_spectra": nearest_spectra,
        "eigenvalue_archive_sha256": sha256(Path("eigenvalues_occupations.npz")),
    }


def main() -> int:
    case_id = os.environ["CASE_ID"].strip()
    lattice_A = float(os.environ["LATTICE_A"])
    started = time.time()
    result = {
        "schema_version": "1.0",
        "candidate": "PdH4",
        "formula_cell": "Pd2H8",
        "stage": 1,
        "gate": "PBE_plane_wave_fixed_cell_volume_and_structural_replication",
        "calculation_id": case_id,
        "lattice_constant_A": lattice_A,
        "status": "RUNNING",
        "scientific_gate_status": "RUNNING",
        "settings": {
            "code": "GPAW",
            "version": importlib.metadata.version("gpaw"),
            "xc": "PBE",
            "mode": "PW",
            "cutoff_eV": CUTOFF_EV,
            "kmesh": list(KMESH),
            "gamma_centered": True,
            "occupation_width_eV": FERMI_WIDTH_EV,
            "spin_polarized": True,
            "initial_Pd_moments_muB": [1.0, 1.0],
            "symmetry": "off",
            "fixed_cell": True,
            "optimizer": "BFGS",
            "fmax_eV_per_A": FMAX_EV_A,
            "max_steps": MAX_STEPS,
            "maxstep_A": MAXSTEP_A,
            "hydrogen_perturbation_A": PERTURBATION_A,
            "perturbation_seed": PERTURBATION_SEED,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "ase": importlib.metadata.version("ase"),
            "gpaw": importlib.metadata.version("gpaw"),
            "numpy": np.__version__,
        },
    }
    try:
        ideal = build_ideal_structure(lattice_A)
        perturbed = ideal.copy()
        displacement = perturb_hydrogen(perturbed)
        atoms = perturbed.copy()
        atoms.set_initial_magnetic_moments([1.0, 1.0] + [0.0] * 8)
        np.save("registered_initial_perturbation_A.npy", displacement)
        write("ideal_structure.vasp", ideal, format="vasp", direct=True, sort=False, vasp5=True)
        write("initial_perturbed_structure.vasp", perturbed, format="vasp", direct=True, sort=False, vasp5=True)
        result["input_hashes"] = {
            "ideal_structure_sha256": sha256(Path("ideal_structure.vasp")),
            "initial_perturbed_structure_sha256": sha256(Path("initial_perturbed_structure.vasp")),
            "perturbation_array_sha256": sha256(Path("registered_initial_perturbation_A.npy")),
        }
        atoms.calc = make_calculator("gpaw-relax.txt")
        initial_energy = float(atoms.get_potential_energy())
        initial_forces = np.asarray(atoms.get_forces(), dtype=float)
        result["initial"] = {
            "energy_eV_per_cell": initial_energy,
            "forces": force_metrics(initial_forces, atoms.get_chemical_symbols()),
            "stress_eV_per_A3": np.asarray(atoms.get_stress(voigt=False), dtype=float).tolist(),
            "geometry": geometry_metrics(atoms),
            "displacement_from_ideal": displacement_metrics(ideal, atoms),
        }
        optimizer = BFGS(atoms, logfile="optimizer.log", trajectory="optimizer.traj", maxstep=MAXSTEP_A)

        def checkpoint() -> None:
            forces = np.asarray(atoms.get_forces(), dtype=float)
            Path("progress.json").write_text(
                json.dumps(
                    {
                        "calculation_id": case_id,
                        "lattice_constant_A": lattice_A,
                        "step": int(optimizer.nsteps),
                        "energy_eV_per_cell": float(atoms.get_potential_energy()),
                        "forces": force_metrics(forces, atoms.get_chemical_symbols()),
                        "geometry": geometry_metrics(atoms),
                        "updated_unix": time.time(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            write("endpoint_current.vasp", atoms, format="vasp", direct=True, sort=False, vasp5=True)

        optimizer.attach(checkpoint, interval=1)
        converged = bool(optimizer.run(fmax=FMAX_EV_A, steps=MAX_STEPS))
        checkpoint()
        final_energy = float(atoms.get_potential_energy())
        final_forces = np.asarray(atoms.get_forces(), dtype=float)
        try:
            total_moment = float(atoms.get_magnetic_moment())
            local_moments = np.asarray(atoms.get_magnetic_moments(), dtype=float)
        except Exception:
            total_moment = float("nan")
            local_moments = np.full(len(atoms), np.nan)
        write("endpoint.vasp", atoms, format="vasp", direct=True, sort=False, vasp5=True)
        np.save("final_forces_eV_A.npy", final_forces)
        np.save("final_local_moments_muB.npy", local_moments)
        atoms.calc.write("state_final.gpw")
        result["optimization"] = {"converged": converged, "steps": int(optimizer.nsteps)}
        result["final"] = {
            "energy_eV_per_cell": final_energy,
            "energy_change_from_initial_eV_per_cell": final_energy - initial_energy,
            "forces": force_metrics(final_forces, atoms.get_chemical_symbols()),
            "stress_eV_per_A3": np.asarray(atoms.get_stress(voigt=False), dtype=float).tolist(),
            "geometry": geometry_metrics(atoms),
            "displacement_from_ideal": displacement_metrics(ideal, atoms),
            "displacement_from_perturbed_input": displacement_metrics(perturbed, atoms),
            "total_magnetic_moment_muB": total_moment,
            "local_magnetic_moments_muB": local_moments.tolist(),
            "max_abs_local_moment_muB": float(np.nanmax(np.abs(local_moments))),
            "endpoint_sha256": sha256(Path("endpoint.vasp")),
            "electronic": electronic_snapshot(atoms.calc),
        }
        result["status"] = "COMPLETED" if converged else "INCONCLUSIVE"
        result["scientific_gate_status"] = "PENDING_CROSS_VOLUME_PARSE"
    except Exception as exc:
        result.update(
            {
                "status": "FAILED",
                "scientific_gate_status": "FAILED_EXECUTION",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        result["wall_time_seconds"] = time.time() - started
        Path("result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n")
    return 0 if result["status"] in {"COMPLETED", "INCONCLUSIVE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
