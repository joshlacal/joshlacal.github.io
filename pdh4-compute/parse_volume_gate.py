from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from ase.io import read

EXPECTED = {"a3p6": 3.6, "a3p8": 3.8, "a4p0": 4.0, "a4p2": 4.2, "a4p4": 4.4}
CRITERIA = {
    "fmax_eV_per_A_max": 0.05,
    "minimum_HH_A_min": 1.0,
    "Pd_H_coordination_within_2p25_A_each": 8,
    "total_moment_abs_muB_max": 0.10,
    "fractionally_occupied_state_count_min": 1,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_manifest(directory: Path) -> dict:
    path = directory / "SHA256SUMS"
    if not path.exists():
        return {"passed": False, "missing": True, "entries": []}
    entries = []
    passed = True
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        target = directory / name
        actual = sha256(target) if target.exists() else None
        ok = actual == expected
        passed &= ok
        entries.append({"file": name, "expected": expected, "actual": actual, "passed": ok})
    return {"passed": passed, "missing": False, "entries": entries}


def last_raw_energy(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    for pattern in (
        r"Extrapolated:\s+([-+0-9.Ee]+)",
        r"Free energy:\s+([-+0-9.Ee]+)",
        r"Potential energy:\s+([-+0-9.Ee]+)",
    ):
        values = re.findall(pattern, text)
        if values:
            return float(values[-1])
    return None


def independent_geometry(path: Path) -> dict:
    atoms = read(path, format="vasp")
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
                "Pd_index": index,
                "nearest_8_H_A": distances[:8],
                "count_H_within_2p25_A": int(sum(distance < 2.25 for distance in distances)),
            }
        )
    passed = (
        len(atoms) == 10
        and symbols.count("Pd") == 2
        and symbols.count("H") == 8
        and hh[0] >= CRITERIA["minimum_HH_A_min"]
        and len(coordination) == 2
        and min(item["count_H_within_2p25_A"] for item in coordination)
        >= CRITERIA["Pd_H_coordination_within_2p25_A_each"]
    )
    return {
        "formula_counts": {symbol: symbols.count(symbol) for symbol in sorted(set(symbols))},
        "minimum_HH_A": float(hh[0]),
        "h2_like_pair_count_lt_1A": int(sum(distance < 1.0 for distance in hh)),
        "Pd_H_coordination": coordination,
        "passed": passed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    report = {
        "schema_version": "1.0",
        "candidate": "PdH4",
        "stage": 1,
        "gate": "PBE_plane_wave_fixed_cell_volume_and_structural_replication",
        "criteria": CRITERIA,
        "status": "RUNNING",
        "cases": {},
    }
    admissible = []
    missing = []
    execution_failures = []
    material_kills = []

    for case_id, lattice_A in EXPECTED.items():
        directory = args.artifact_root / case_id
        item = {"case_id": case_id, "lattice_constant_A_expected": lattice_A, "manifest": verify_manifest(directory)}
        result_path = directory / "result.json"
        endpoint_path = directory / "endpoint.vasp"
        force_path = directory / "final_forces_eV_A.npy"
        if not result_path.exists():
            item.update({"status": "MISSING", "admissible": False})
            missing.append(case_id)
            report["cases"][case_id] = item
            continue
        result = json.loads(result_path.read_text())
        item["result_json_sha256"] = sha256(result_path)
        item["result"] = result
        if result.get("status") == "FAILED" or not endpoint_path.exists() or not force_path.exists():
            item.update({"status": "FAILED_EXECUTION", "admissible": False})
            execution_failures.append(case_id)
            report["cases"][case_id] = item
            continue
        forces = np.asarray(np.load(force_path), dtype=float)
        norms = np.linalg.norm(forces, axis=1)
        independent_fmax = float(norms.max())
        independent_frms = float(np.sqrt(np.mean(norms**2)))
        final = result.get("final", {})
        energy = float(final.get("energy_eV_per_cell", np.nan))
        reported_fmax = float(final.get("forces", {}).get("max_force_eV_per_A", np.nan))
        raw_energy = last_raw_energy(directory / "gpaw-relax.txt")
        geometry = independent_geometry(endpoint_path)
        total_moment = abs(float(final.get("total_magnetic_moment_muB", np.inf)))
        fractional = int(final.get("electronic", {}).get("fractionally_occupied_state_count", 0))
        topology_kill = not geometry["passed"]
        magnetic_kill = total_moment > CRITERIA["total_moment_abs_muB_max"]
        insulating_kill = fractional < CRITERIA["fractionally_occupied_state_count_min"]
        item.update(
            {
                "status": result.get("status"),
                "energy_eV_per_cell": energy,
                "raw_gpaw_energy_eV_per_cell": raw_energy,
                "raw_energy_agrees": raw_energy is not None and abs(raw_energy - energy) <= 2.0e-4,
                "independent_fmax_eV_per_A": independent_fmax,
                "independent_force_rms_eV_per_A": independent_frms,
                "force_array_agrees": abs(independent_fmax - reported_fmax) <= 1.0e-10,
                "geometry": geometry,
                "total_moment_abs_muB": total_moment,
                "fractionally_occupied_state_count": fractional,
                "endpoint_sha256": sha256(endpoint_path),
                "topology_kill": topology_kill,
                "magnetic_kill": magnetic_kill,
                "insulating_kill": insulating_kill,
            }
        )
        for active, trigger in ((topology_kill, "topology"), (magnetic_kill, "magnetism"), (insulating_kill, "insulating")):
            if active:
                material_kills.append({"case": case_id, "trigger": trigger})
        item["admissible"] = all(
            [
                item["manifest"]["passed"],
                result.get("status") == "COMPLETED",
                bool(result.get("optimization", {}).get("converged")),
                independent_fmax <= CRITERIA["fmax_eV_per_A_max"],
                item["force_array_agrees"],
                item["raw_energy_agrees"],
                geometry["passed"],
                not magnetic_kill,
                not insulating_kill,
            ]
        )
        if item["admissible"]:
            admissible.append(item)
        report["cases"][case_id] = item

    report["missing_cases"] = missing
    report["execution_failures"] = execution_failures
    report["material_kill_triggers"] = material_kills
    report["admissible_case_count"] = len(admissible)
    if material_kills:
        report["status"] = "KILLED_MATERIAL"
        report["decision"] = "Reject PdH4 under the first registered material trigger."
    elif missing:
        report["status"] = "RUNNING"
        report["decision"] = "Wait for terminal artifacts; no partial result is admissible."
    elif execution_failures:
        report["status"] = "FAILED_EXECUTION"
        report["decision"] = "Version and repair only the documented execution failure."
    elif len(admissible) != len(EXPECTED):
        report["status"] = "KILLED_NONSTATIONARY"
        report["decision"] = "One or more registered volumes failed the stationary-endpoint criterion."
    else:
        ordered = sorted(admissible, key=lambda item: item["lattice_constant_A_expected"])
        lattices = np.asarray([item["lattice_constant_A_expected"] for item in ordered])
        energies = np.asarray([item["energy_eV_per_cell"] for item in ordered])
        minimum_index = int(np.argmin(energies))
        report["energy_curve"] = [
            {"lattice_constant_A": float(a), "energy_eV_per_cell": float(e), "delta_E_from_min_eV_per_cell": float(e - energies.min())}
            for a, e in zip(lattices, energies)
        ]
        report["minimum_sampled"] = report["energy_curve"][minimum_index]
        report["interior_minimum"] = bool(0 < minimum_index < len(ordered) - 1)
        if report["interior_minimum"]:
            sl = slice(minimum_index - 1, minimum_index + 2)
            coefficients = np.polyfit(lattices[sl], energies[sl], 2)
            vertex = -coefficients[1] / (2.0 * coefficients[0])
            report["local_quadratic_fit"] = {
                "coefficients": coefficients.tolist(),
                "vertex_A": float(vertex),
                "positive_curvature": bool(coefficients[0] > 0.0),
            }
            if coefficients[0] > 0.0 and lattices[0] < vertex < lattices[-1]:
                report["status"] = "PASSED_VOLUME_GATE"
                report["decision"] = "Proceed only to endpoint replication and targeted Pd-H decomposition screening."
            else:
                report["status"] = "KILLED_NO_BOUNDED_MINIMUM"
                report["decision"] = "The apparent interior point is not a bounded positive-curvature minimum."
        else:
            report["status"] = "EXPAND_LOW" if minimum_index == 0 else "EXPAND_HIGH"
            report["decision"] = "Expand once only in the downhill direction."

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0 if report["status"] in {"RUNNING", "PASSED_VOLUME_GATE", "EXPAND_LOW", "EXPAND_HIGH"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
