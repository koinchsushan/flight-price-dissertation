"""Check that the environment is correctly set up before running the notebooks.

Run this after following the setup steps in the README:

    python scripts/verify_setup.py

Every check prints OK or FAIL with the exact remedy, so a failure tells you what
to do rather than only that something is wrong. Exits 0 if everything passes.
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

# (import name, pip name, why it is needed)
REQUIRED = [
    ("pandas", "pandas", "data handling"),
    ("numpy", "numpy", "numerics"),
    ("pyarrow", "pyarrow", "Parquet read/write"),
    ("scipy", "scipy", "statistics"),
    ("sklearn", "scikit-learn", "metrics and baselines"),
    ("statsmodels", "statsmodels", "SARIMA / SARIMAX"),
    ("xgboost", "xgboost", "gradient-boosted trees"),
    ("torch", "torch", "LSTM"),
    ("imblearn", "imbalanced-learn", "class-imbalance utilities"),
    ("matplotlib", "matplotlib", "figures"),
    ("seaborn", "seaborn", "figures"),
]

IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"

failures: list[str] = []


def ok(msg: str) -> None:
    print(f"  OK    {msg}")


def fail(msg: str, remedy: str) -> None:
    print(f"  FAIL  {msg}")
    print(f"        -> {remedy}")
    failures.append(msg)


def check_python() -> None:
    print("Python")
    major, minor = sys.version_info[:2]
    version = f"{major}.{minor}.{sys.version_info[2]}"
    if (major, minor) >= (3, 12):
        ok(f"version {version}")
    else:
        fail(
            f"version {version} is too old",
            "install Python 3.12 or newer, then recreate the virtual environment",
        )

    # Running inside a venv? sys.prefix differs from base_prefix when active.
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        ok(f"running inside a virtual environment ({Path(sys.prefix).name})")
    else:
        fail(
            "not running inside a virtual environment",
            "activate it first: "
            + (r".venv\Scripts\activate" if IS_WINDOWS else "source .venv/bin/activate"),
        )


def check_packages() -> None:
    print("\nPackages")
    for module_name, pip_name, purpose in REQUIRED:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            fail(f"{pip_name} not installed ({purpose})", f"pip install {pip_name}")
            continue
        except Exception as exc:  # noqa: BLE001 - we want the real reason shown
            # xgboost imports but fails to load its shared library without OpenMP.
            if module_name == "xgboost" and "libomp" in str(exc).lower():
                remedy = (
                    "brew install libomp"
                    if IS_MACOS
                    else "install the OpenMP runtime for your platform"
                )
                fail(f"{pip_name} installed but cannot load (OpenMP missing)", remedy)
            else:
                fail(f"{pip_name} installed but failed to import: {exc}", "see the error above")
            continue

        version = getattr(module, "__version__", "unknown")
        ok(f"{pip_name} {version}")


def check_torch_device() -> None:
    print("\nPyTorch compute device")
    try:
        import torch
    except Exception:  # noqa: BLE001
        print("  SKIP  torch unavailable (reported above)")
        return

    if torch.cuda.is_available():
        ok(f"CUDA available - {torch.cuda.get_device_name(0)}")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        ok("Apple MPS available - the LSTM will train on the GPU")
    else:
        ok("CPU only - everything still runs, the LSTM will simply be slower")


def check_project_package() -> None:
    print("\nProject package")
    try:
        from flightprice.config import PROJECT_ROOT, RANDOM_SEED, SPIKE_WINDOW
    except ImportError:
        fail(
            "'flightprice' is not importable",
            "run 'pip install -e .' from the repository root",
        )
        return

    ok(f"flightprice importable (seed {RANDOM_SEED}, spike window {SPIKE_WINDOW})")
    ok(f"project root resolved to {PROJECT_ROOT}")


def check_data() -> None:
    print("\nData")
    try:
        from flightprice.config import PROCESSED_DIR, RAW_SUBSET
    except ImportError:
        print("  SKIP  cannot locate data paths until the package imports")
        return

    if RAW_SUBSET.exists():
        size_mb = RAW_SUBSET.stat().st_size / 1e6
        ok(f"{RAW_SUBSET.name} present ({size_mb:,.0f} MB)")
    else:
        fail(
            f"{RAW_SUBSET.name} not found in {RAW_SUBSET.parent}",
            "see data/README.md - download it, or rebuild it with scripts/build_subset.py",
        )

    processed = sorted(PROCESSED_DIR.glob("*.parquet")) if PROCESSED_DIR.exists() else []
    if processed:
        ok(f"{len(processed)} processed file(s): {', '.join(p.name for p in processed)}")
    else:
        print("  NOTE  no processed files yet - notebook 01 creates them")


def check_kernel() -> None:
    print("\nJupyter kernel")
    try:
        from jupyter_client.kernelspec import KernelSpecManager
    except ImportError:
        print("  SKIP  jupyter_client not installed")
        return

    names = KernelSpecManager().find_kernel_specs()
    if "flightprice" in names:
        ok("'flightprice' kernel registered")
    else:
        fail(
            "'flightprice' kernel not registered",
            'python -m ipykernel install --user --name flightprice '
            '--display-name "Python (flightprice)"',
        )


def main() -> int:
    print(f"Environment check - {platform.system()} {platform.machine()}\n")
    check_python()
    check_packages()
    check_torch_device()
    check_project_package()
    check_data()
    check_kernel()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed. Fix the items marked FAIL above, then re-run.")
        return 1

    print("All checks passed. Open notebooks/01_data_cleaning.ipynb to begin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
