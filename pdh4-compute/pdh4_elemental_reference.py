from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import write
from gpaw import FermiDirac, GPAW, Mixer, MixerSum, PW

CUTOFF_EV = 600.0
FERMI_WIDTH_EV = 0.05


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def make_pd(lattice_A: float) -> Atoms:
    atoms = bulk("Pd", "fcc", a=lattice_A, cubic=True)
    atoms.pbc = True
    atoms.set_initial_magnetic_moments([1.0] * len(atoms))
    return atoms


def make_h2(bond_A: float) -> Atoms:
    cell = 15.0
    center = cell / 2.0
    atoms = Atoms(
        "H2",
        positions=[[center - bond_A / 2.0, center, center], [center + bond_A / 2.0, center, center]],
        cell=[cell, cell, cell],
        pbc=True,
    )
    return atoms


def main() -> int:
    case_id = os.environ["CASE_ID"].strip()
    kind = os.environ["KIND"].strip()
    value = float(os.environ["VALUE"])
    started = time.time()
    result = {
        "schema_version": "1.0",
        "candidate_context": "PdH4",
        "stage": "elemental_decomposition_reference",
        "calculation_id": case_id,
        "kind": kind,
        "value": value,
        "status": "RUNNING",
        "settings": {
            "code": "GPAW",
            "version": importlib.metadata.version("gpaw"),
            "xc": "PBE",
            "mode": "PW",
            "cutoff_eV": CUTOFF_EV,
            "occupation_width_eV": FERMI_WIDTH_EV,
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
        if kind == "Pd_fcc":
            atoms = make_pd(value)
            calc = GPAW(
                mode=PW(CUTOFF_EV),
                xc="PBE",
                kpts={"size": (8, 8, 8), "gamma": True},
                occupations=FermiDirac(FERMI_WIDTH_EV),
                spinpol=True,
                mixer=MixerSum(beta=0.05, nmaxold=5, weight=50.0),
                symmetry="off",
                maxiter=400,
                convergence={"energy": 1.0e-5, "density": 2.0e-4, "eigenstates": 2.0e-6},
                txt="gpaw.txt",
            )
            atoms.calc = calc
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
            stress = np.asarray(atoms.get_stress(voigt=False), dtype=float)
            total_moment = float(atoms.get_magnetic_moment())
            local_moments = np.asarray(atoms.get_magnetic_moments(), dtype=float)
            result.update(
                {
                    "structure": "conventional cubic fcc Pd",
                    "atom_count": len(atoms),
                    "lattice_constant_A": value,
                    "kmesh": [8, 8, 8],
                    "spin_polarized": True,
                    "energy_eV_per_cell": energy,
                    "energy_eV_per_Pd_atom": energy / len(atoms),
                    "max_force_eV_per_A": float(np.linalg.norm(forces, axis=1).max()),
                    "stress_eV_per_A3": stress.tolist(),
                    "total_magnetic_moment_muB": total_moment,
                    "local_magnetic_moments_muB": local_moments.tolist(),
                }
            )
        elif kind == "H2":
            atoms = make_h2(value)
            calc = GPAW(
                mode=PW(CUTOFF_EV),
                xc="PBE",
                kpts=(1, 1, 1),
                occupations=FermiDirac(FERMI_WIDTH_EV),
                spinpol=False,
                mixer=Mixer(beta=0.05, nmaxold=5, weight=50.0),
                symmetry="off",
                maxiter=400,
                convergence={"energy": 1.0e-6, "density": 1.0e-5, "eigenstates": 1.0e-7},
                txt="gpaw.txt",
            )
            atoms.calc = calc
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
            result.update(
                {
                    "structure": "isolated H2 in 15 A cubic cell",
                    "atom_count": 2,
                    "bond_length_A": value,
                    "kmesh": [1, 1, 1],
                    "spin_polarized": False,
                    "energy_eV_per_H2": energy,
                    "max_force_eV_per_A": float(np.linalg.norm(forces, axis=1).max()),
                    "force_vectors_eV_per_A": forces.tolist(),
                }
            )
        else:
            raise ValueError(f"Unsupported reference kind: {kind}")
        write("structure.vasp", atoms, format="vasp", direct=False, sort=False, vasp5=True)
        np.save("forces_eV_A.npy", forces)
        calc.write("state.gpw")
        result["structure_sha256"] = sha256(Path("structure.vasp"))
        result["status"] = "COMPLETED"
    except Exception as exc:
        result.update(
            {
                "status": "FAILED",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        result["wall_time_seconds"] = time.time() - started
        Path("result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n")
    return 0 if result["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
