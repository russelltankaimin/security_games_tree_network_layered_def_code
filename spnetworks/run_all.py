"""Run the full (small, fast) experiment suite and regenerate all figures."""
import exp1_validation, exp2_scalability, exp3_baselines, exp4_nesting, exp5_sensitivity, plots

def main():
    exp1_validation.run()
    exp2_scalability.run()
    exp3_baselines.run()
    exp4_nesting.run()
    exp5_sensitivity.run()
    plots.all_figs()
    print("done -- CSVs in results/, figures in figures/")

if __name__ == "__main__":
    main()
