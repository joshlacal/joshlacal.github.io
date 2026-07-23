from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

EXPECTED = {
    "pd_a3p7": ("Pd_fcc", 3.7),
    "pd_a3p8": ("Pd_fcc", 3.8),
    "pd_a3p9": ("Pd_fcc", 3.9),
    "pd_a4p0": ("Pd_fcc", 4.0),
    "pd_a4p1": ("Pd_fcc", 4.1),
    "h2_b0p70": ("H2", 0.70),
    "h2_b0p74": ("H2", 0.74),
    "h2_b0p78": ("H2", 0.78),
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
        return {"passed": False, "entries": [], "missing": True}
    passed = True
    entries = []
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
    return {"passed": passed, "entries": entries, "missing": False}


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = {
        "schema_version": "1.0",
        "candidate_context": "PdH4",
        "gate": "elemental_decomposition_references",
        "status": "RUNNING",
        "cases": {},
    }
    missing = []
    failed = []
    pd_points = []
    h2_points = []
    for case_id, (kind, expected_value) in EXPECTED.items():
        directory = args.artifact_root / case_id
        item = {"case_id": case_id, "expected_kind": kind, "expected_value": expected_value, "manifest": verify_manifest(directory)}
        result_path = directory / "result.json"
        if not result_path.exists():
            item.update({"status": "MISSING", "passed": False})
            missing.append(case_id)
            report["cases"][case_id] = item
            continue
        result = json.loads(result_path.read_text())
        raw_energy = last_raw_energy(directory / "gpaw.txt")
        if kind == "Pd_fcc":
            energy = float(result.get("energy_eV_per_cell", np.nan))
            normalized = float(result.get("energy_eV_per_Pd_atom", np.nan))
            normalization_ok = abs(normalized * 4.0 - energy) <= 1.0e-10
        else:
            energy = float(result.get("energy_eV_per_H2", np.nan))
            normalized = energy
            normalization_ok = True
        item.update(
            {
                "status": result.get("status"),
                "result_json_sha256": sha256(result_path),
                "reported_energy_eV": energy,
                "normalized_reference_energy_eV": normalized,
                "raw_gpaw_energy_eV": raw_energy,
                "raw_energy_agrees": raw_energy is not None and abs(raw_energy - energy) <= 2.0e-4,
                "normalization_ok": normalization_ok,
                "kind_matches": result.get("kind") == kind,
                "value_matches": abs(float(result.get("value", np.inf)) - expected_value) <= 1.0e-12,
            }
        )
        item["passed"] = all(
            [
                item["manifest"]["passed"],
                result.get("status") == "COMPLETED",
                item["raw_energy_agrees"],
                normalization_ok,
                item["kind_matches"],
                item["value_matches"],
                np.isfinite(normalized),
            ]
        )
        if not item["passed"]:
            failed.append(case_id)
        elif kind == "Pd_fcc":
            pd_points.append((expected_value, normalized, case_id))
        else:
            h2_points.append((expected_value, normalized, case_id))
        report["cases"][case_id] = item

    report["missing_cases"] = missing
    report["failed_cases"] = failed
    if missing:
        report["status"] = "RUNNING"
        report["decision"] = "Wait for all reference artifacts."
    elif failed:
        report["status"] = "FAILED_EXECUTION"
        report["decision"] = "Repair only the documented failed reference calculation."
    else:
        pd_points.sort()
        h2_points.sort()
        pd_a = np.asarray([point[0] for point in pd_points])
        pd_e = np.asarray([point[1] for point in pd_points])
        h2_b = np.asarray([point[0] for point in h2_points])
        h2_e = np.asarray([point[1] for point in h2_points])
        pd_min = int(np.argmin(pd_e))
        h2_min = int(np.argmin(h2_e))
        report["Pd_fcc_points"] = [
            {"lattice_constant_A": float(a), "energy_eV_per_Pd_atom": float(e), "case_id": case}
            for a, e, case in pd_points
        ]
        report["H2_points"] = [
            {"bond_length_A": float(b), "energy_eV_per_H2": float(e), "case_id": case}
            for b, e, case in h2_points
        ]
        report["Pd_fcc_minimum_sampled"] = report["Pd_fcc_points"][pd_min]
        report["H2_minimum_sampled"] = report["H2_points"][h2_min]
        pd_interior = 0 < pd_min < len(pd_points) - 1
        h2_interior = 0 < h2_min < len(h2_points) - 1
        report["Pd_fcc_interior_minimum"] = pd_interior
        report["H2_interior_minimum"] = h2_interior
        if pd_interior:
            sl = slice(pd_min - 1, pd_min + 2)
            coeff = np.polyfit(pd_a[sl], pd_e[sl], 2)
            report["Pd_fcc_quadratic"] = {
                "coefficients": coeff.tolist(),
                "vertex_A": float(-coeff[1] / (2.0 * coeff[0])),
                "positive_curvature": bool(coeff[0] > 0.0),
            }
        if h2_interior:
            coeff = np.polyfit(h2_b, h2_e, 2)
            report["H2_quadratic"] = {
                "coefficients": coeff.tolist(),
                "vertex_A": float(-coeff[1] / (2.0 * coeff[0])),
                "minimum_energy_eV_per_H2": float(np.polyval(coeff, -coeff[1] / (2.0 * coeff[0]))),
                "positive_curvature": bool(coeff[0] > 0.0),
            }
        if not pd_interior or not h2_interior:
            report["status"] = "EXPAND_REFERENCE_BRACKET"
            report["decision"] = "Expand only the edge-pinned reference bracket."
        elif not report["Pd_fcc_quadratic"]["positive_curvature"] or not report["H2_quadratic"]["positive_curvature"]:
            report["status"] = "FAILED_REFERENCE_FIT"
            report["decision"] = "Reference energy curve lacks positive curvature."
        else:
            report["status"] = "PASSED_REFERENCES"
            report["decision"] = "Use these independently parsed reference energies only after a PdH4 structural endpoint passes."

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0 if report["status"] in {"RUNNING", "PASSED_REFERENCES", "EXPAND_REFERENCE_BRACKET"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
