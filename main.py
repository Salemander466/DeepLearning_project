# ============================================================
# main.py
# Run either:
# 1. Hyperparameter tuning
# 2. Full Xtrain training + Xtest evaluation
# ============================================================

import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path("/Users/jacobbae/Documents/UU25/DeepLearning_project")

# Change these filenames if your scripts have different names.
HYPERPARAM_SCRIPT = PROJECT_DIR / "hyperpram_tuning.py"
FULL_TRAIN_SCRIPT = PROJECT_DIR / "single_run_swa.py"

#The code used to run the hyperparameter tuning and the final training + evaluation. 
def run_script(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find script: {script_path}")

    print("\n" + "=" * 80)
    print(f"Running: {script_path.name}")
    print("=" * 80 + "\n")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_DIR),
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_path.name} failed with exit code {result.returncode}"
        )

    print("\n" + "=" * 80)
    print(f"Finished: {script_path.name}")
    print("=" * 80 + "\n")


def main():
    print("\nChoose what to run:")
    print("1 = Hyperparameter tuning")
    print("2 = Train final model on full Xtrain and test on Xtest")
    print("q = Quit")

    choice = input("\nEnter choice: ").strip().lower()

    if choice == "1":
        run_script(HYPERPARAM_SCRIPT)

    elif choice == "2":
        run_script(FULL_TRAIN_SCRIPT)

    elif choice in {"q", "quit", "exit"}:
        print("Exiting.")

    else:
        print(f"Invalid choice: {choice}")
        print("Use 1, 2, or q.")


if __name__ == "__main__":
    main()