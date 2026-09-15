"""Run the earlier five-model comparison saved in results/research/."""

from src.research_analysis import run_analysis
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/research')
    args = parser.parse_args()
    summary, importance, metadata = run_analysis(
        data_path=ROOT / 'data/prostate.csv', output_dir=args.output_dir
    )
    print("\nRepeated cross-validation summary")
    print(summary.to_string(index=False))
    print(f"\nBest model by mean held-out RMSE: {metadata['best_model']}")
    print("\nHeld-out permutation importance")
    print(importance.to_string(index=False))
    print(f"\nOutputs saved in {args.output_dir}")
    print("These are pilot results for log-PSA prediction, not clinical validation.")


if __name__ == "__main__":
    main()
