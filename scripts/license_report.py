"""Print name | version | license for the direct dependencies (task 2.2).

Reads the installed package metadata (License-Expression, then License, then
license classifiers), so each row is a package-manager query on this machine.
Run with the project environment's Python:
    ~/.venvs/learned-opponents/bin/python scripts/license_report.py
"""

from __future__ import annotations

from importlib import metadata

DIRECT = ["numpy", "pyyaml", "onnx", "onnxruntime", "matplotlib", "pytest", "hypothesis",
          "mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch", "rsl-rl-lib", "tensordict",
          "tensorboard"]


def license_of(dist: metadata.Distribution) -> str:
    md = dist.metadata
    expr = md.get("License-Expression")
    if expr:
        return expr.strip()
    lic = (md.get("License") or "").strip()
    if lic and len(lic) < 80 and "\n" not in lic:
        return lic
    classifiers = [c.split("::")[-1].strip() for c in md.get_all("Classifier") or []
                   if c.startswith("License ::")]
    if classifiers:
        return "; ".join(classifiers)
    return (lic.splitlines()[0][:80] + " (first line of License field)") if lic else "UNKNOWN"


def main() -> None:
    print("| Name | Version | License (installed metadata) |")
    print("| --- | --- | --- |")
    for name in DIRECT:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            print(f"| {name} | not installed | - |")
            continue
        print(f"| {name} | {dist.version} | {license_of(dist)} |")


if __name__ == "__main__":
    main()
