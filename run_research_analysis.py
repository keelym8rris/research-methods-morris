"""Run the thesis-grade Part 1 analysis."""

from src.research_analysis import run_analysis


def main():
    summary, importance, metadata = run_analysis()
    print("\nRepeated cross-validation summary")
    print(summary.to_string(index=False))
    print(f"\nBest model by mean held-out RMSE: {metadata['best_model']}")
    print("\nHeld-out permutation importance")
    print(importance.to_string(index=False))
    print("\nOutputs saved in results/research/")
    print("These are pilot results for log-PSA prediction, not clinical validation.")


if __name__ == "__main__":
    main()
