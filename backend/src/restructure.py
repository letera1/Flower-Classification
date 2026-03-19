#!/usr/bin/env python3
"""Setup professional ML project structure."""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
BACKEND = ROOT / "backend"

# Create modern ML project structure
dirs = [
    "backend/app",
    "backend/app/api",
    "backend/app/core",
    "backend/app/models",
    "backend/app/schemas",
    "backend/app/services",
    "backend/app/utils",
    "backend/src",
    "backend/src/data",
    "backend/src/features",
    "backend/src/models",
    "backend/src/evaluation",
    "backend/tests",
    "backend/tests/unit",
    "backend/tests/integration",
    "backend/notebooks",
    "backend/data/raw",
    "backend/data/processed",
    "backend/models/artifacts",
    "backend/logs",
    "backend/configs",
    "backend/scripts",
    "backend/docker",
]

for d in dirs:
    (ROOT / d).mkdir(parents=True, exist_ok=True)
    # Add __init__.py to Python packages
    if any(x in d for x in ["app", "src", "tests"]):
        (ROOT / d / "__init__.py").touch()

# Add __init__.py to app subdirectories
for sub in ["api", "core", "models", "schemas", "services", "utils"]:
    (ROOT / f"backend/app/{sub}/__init__.py").touch()

# Add __init__.py to src subdirectories
for sub in ["data", "features", "models", "evaluation"]:
    (ROOT / f"backend/src/{sub}/__init__.py").touch()

print("Created directory structure")

# Move existing files
files_to_move = [
    ("backend/app.py", "backend/app/main.py"),
    ("cli/predict.py", "backend/scripts/cli.py"),
    ("train_pipeline.py", "backend/scripts/train.py"),
    ("notebooks/01_eda_flower_classification.ipynb", "backend/notebooks/01_eda_flower_classification.ipynb"),
    ("src/data/preprocessor.py", "backend/src/data/preprocessor.py"),
    ("src/data/feature_engineering.py", "backend/src/features/engineering.py"),
    ("src/models/trainer.py", "backend/src/models/trainer.py"),
    ("src/evaluation/evaluator.py", "backend/src/evaluation/evaluator.py"),
    ("src/utils/helpers.py", "backend/app/utils/helpers.py"),
]

for src, dst in files_to_move:
    src_path = ROOT / src
    dst_path = ROOT / dst
    if src_path.exists():
        shutil.move(str(src_path), str(dst_path))
        print(f"Moved {src} -> {dst}")

# Create .gitkeep files
for keep_dir in ["backend/data/raw", "backend/data/processed", "backend/models/artifacts", "backend/logs"]:
    (ROOT / f"{keep_dir}/.gitkeep").touch()

print("\nStructure setup complete!")
