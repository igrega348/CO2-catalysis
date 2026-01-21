"""
Manual triplet testing for active learning experiments.
This script allows manual entry of triplet combinations from the older code
to verify if the refactored code reproduces the same metrics.

Usage:
    1. Modify the MANUAL_TRIPLETS dictionary below to add triplet combinations
    2. Run: python G_M.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from carbondriver import GDEOptimizer
from carbondriver.loaders import load_gas_data

# ============================================================================
# CONFIGURATION - Modify these to test specific triplet combinations
# ============================================================================

# Manual triplets to test (from early_peak_summary.csv)
# Format: {model_name: [[triplet1, triplet2, triplet3], ...]}
# Examples from the older code that performed well:
MANUAL_TRIPLETS = {
    'Ph': [
        # From Ph_F28 (peak_step=0, fastest)
        [9, 28, 27],
        # From Ph_F13 (peak_step=2)
        [24, 29, 20],
        # From Ph_F25 (peak_step=2)
        [28, 24, 11],
        # From Ph_F27 (peak_step=2)
        [17, 20, 9],
    ],
    'GP+Ph': [
        # From GP_Ph_F12 (peak_step=-1, found triplet 27 immediately)
        [5, 27, 18],
        # From GP_Ph_F23 (peak_step=-2)
        [27, 21, 26],
        # From GP_Ph_F1 (peak_step=2)
        [22, 4, 17],
        # From GP_Ph_F57 (peak_step=2)
        [16, 6, 3],
        # Run from GP_Ph_F94 (peak_step=2)
        [7, 9, 20]
    ],
}

# Models to test
MODELS_TO_TEST = ['Ph']  # Add 'GP+Ph' if needed

# Output directory
OUTPUT_BASE = Path('./active_learning_results_manual')
OUTPUT_BASE.mkdir(exist_ok=True)

# Load data once
df = load_gas_data("paper/Characterization_data.xlsx")


def run_manual_experiment(model_name: str, initial_triplets: list, run_idx: int):
    """
    Run a single active learning experiment with manually specified initial triplets.
    
    Args:
        model_name: Name of the model ('Ph', 'GP+Ph', etc.)
        initial_triplets: List of 3 triplet IDs to use as initial training data
        run_idx: Index for this run (for directory naming)
    
    Returns:
        Path to the run directory
    """
    # Reset df for this run
    df_triplet_means = df.groupby('triplet').mean()
    
    run_dir = OUTPUT_BASE / f'{model_name}_manual_{run_idx:03d}'
    run_dir.mkdir(exist_ok=True, parents=True)
    
    # Save the manual triplets used
    with open(run_dir / 'initial_triplets.txt', 'w') as f:
        f.write(f"Initial triplets: {initial_triplets}\n")
    
    # Initialize optimizer
    gde = GDEOptimizer(
        model_name=model_name,
        aquisition="EI",
        quantity="FE (Eth)",
        maximize=True,
        output_dir=str(run_dir),
        config={'num_iter': 101, 'make_plots': False, 'normalize': True}
    )
    
    # Use manually specified initial triplets
    chosen_triplet_ids = list(initial_triplets)
    
    # Track results
    chosen_triplets_list = chosen_triplet_ids.copy()
    expected_improvements = [None] * len(chosen_triplet_ids)
    
    # Get rows for chosen triplets
    train_df = df[df['triplet'].isin(chosen_triplet_ids)].copy()
    
    # Get rows for withheld triplets
    withheld_df = df_triplet_means[~df_triplet_means.index.isin(chosen_triplet_ids)].copy()

    # Active learning loop
    iteration = 0
    
    while len(withheld_df) > 0:
        # Evaluate acquisition function
        best_ei, best_row_idx = gde.step_within_data(train_df, withheld_df)
        best_triplet = withheld_df.iloc[int(best_row_idx)]
        
        # This line ensures that we append the new triplet data to train_df, not replacing it
        train_df = df[df['triplet'] == best_triplet.name]
        withheld_df = withheld_df.drop(index=best_triplet.name)
        
        expected_improvements.append(float(best_ei))
        chosen_triplets_list.append(best_triplet.name)
        
        # Print the first few selections for debugging
        if iteration < 3:
            print(f"    Iteration {iteration + 1}: Selected triplet {best_triplet.name} (EI: {best_ei:.6f})")
        
        iteration += 1

    # Save results
    results_df = pd.DataFrame({
        'chosen_triplets': chosen_triplets_list,
        'expected_improvements': expected_improvements
    })
    results_df.to_csv(run_dir / 'chosen_triplets.csv')
    
    # Append summary to log file in append mode
    log_file = OUTPUT_BASE / 'runs_summary.txt'
    with open(log_file, 'a') as f:
        f.write(f"Run: {run_dir.stem}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Initial triplets: {initial_triplets}\n")
        f.write(f"4th triplet selected: {chosen_triplets_list[3] if len(chosen_triplets_list) > 3 else 'N/A'}\n")
        f.write(f"Total triplets selected: {len(chosen_triplets_list)}\n")
        f.write("-" * 70 + "\n")
    
    print(f"    4th triplet selected: {chosen_triplets_list[3] if len(chosen_triplets_list) > 3 else 'N/A'}")
    
    return run_dir


def process_runs_mean(model_name: str, output_base: Path):
    """Process all runs for a given model and return aggregated DataFrame."""
    all_df = []
    
    for run_dir in output_base.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith(model_name):
            continue
        
        results_file = run_dir / 'chosen_triplets.csv'
        if not results_file.exists():
            continue
        
        # Read the results
        chosen_df = pd.read_csv(results_file, index_col=0)
        df_triplet_means = df.groupby('triplet').mean()
        
        # Calculate cummax FE for this run
        chosen_df['cummax FE'] = df_triplet_means.loc[
            chosen_df['chosen_triplets'], 'FE (Eth)'
        ].cummax().values
        
        # Add step column (offset by 2 to match old convention)
        i0 = 2
        chosen_df['step'] = chosen_df.index - i0
        chosen_df['dname'] = run_dir.stem
        chosen_df['model'] = model_name
        
        # Read initial triplets if available
        triplet_file = run_dir / 'initial_triplets.txt'
        if triplet_file.exists():
            chosen_df['initial_triplets'] = triplet_file.read_text().strip()
        
        all_df.append(chosen_df)
    
    return pd.concat(all_df, axis=0) if all_df else pd.DataFrame()


def calculate_metrics(model_name: str, output_base: Path, threshold: float = 0.245):
    """Calculate acceleration metrics for a model."""
    steps_to_finish = []
    run_details = []
    
    df_triplet_means = df.groupby('triplet').mean()
    
    for run_dir in output_base.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith(model_name):
            continue
        
        results_file = run_dir / 'chosen_triplets.csv'
        if not results_file.exists():
            continue
        
        chosen_df = pd.read_csv(results_file, index_col=0)
        chosen_df['cummax FE'] = df_triplet_means.loc[
            chosen_df['chosen_triplets'], 'FE (Eth)'
        ].cummax().values
        
        # Add step column (offset by 2 due to triplet indexing)
        i0 = 2
        chosen_df['step'] = chosen_df.index - i0
        
        # Find step where cummax FE exceeds threshold
        filtered = chosen_df[chosen_df['cummax FE'] > threshold]
        if not filtered.empty:
            step_to_max = filtered['step'].iloc[0]
            steps_to_finish.append(max(0, step_to_max))
            
            # Get initial triplets
            triplet_file = run_dir / 'initial_triplets.txt'
            initial = triplet_file.read_text().strip() if triplet_file.exists() else 'Unknown'
            run_details.append({
                'run': run_dir.stem,
                'step_to_threshold': step_to_max,
                'initial_triplets': initial
            })
    
    return steps_to_finish, run_details


def compare_with_reference(reference_csv: str = 'early_peak_summary.csv'):
    """Load reference data from the older code for comparison."""
    ref_df = pd.read_csv(reference_csv)
    
    print("\n" + "="*70)
    print("REFERENCE DATA FROM OLDER CODE (early_peak_summary.csv)")
    print("="*70)
    
    for model_name in ['Ph_F', 'GP_Ph_F']:
        model_data = ref_df[ref_df['model'] == model_name]
        if model_data.empty:
            continue
        
        peak_steps = model_data['peak_step'].values
        mean_step = np.mean(peak_steps)
        std_step = np.std(peak_steps)
        
        print(f"\n{model_name}:")
        print(f"  Mean peak step: {mean_step:.2f} ± {std_step:.2f}")
        print(f"  Best runs (step <= 2):")
        
        best_runs = model_data[model_data['peak_step'] <= 2].head(5)
        for _, row in best_runs.iterrows():
            print(f"    {row['run']}: step={row['peak_step']}, triplets=[{row['early_triplets']}]")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    # Show reference data for comparison
    compare_with_reference()
    
    print("\n" + "="*70)
    print("RUNNING MANUAL TRIPLET EXPERIMENTS")
    print("="*70)
    
    # Run experiments with manual triplets
    for model in MODELS_TO_TEST:
        if model not in MANUAL_TRIPLETS:
            print(f"No manual triplets defined for {model}, skipping...")
            continue
        
        triplet_sets = MANUAL_TRIPLETS[model]
        print(f"\nRunning {len(triplet_sets)} experiments for {model}...")
        
        for run_idx, triplets in enumerate(triplet_sets):
            print(f"\n{'='*60}")
            print(f"STARTING RUN {run_idx} ({model})")
            print(f"Initial triplets: {triplets}")
            print(f"{'='*60}")
            
            run_manual_experiment(model, triplets, run_idx)
            print(f"  {model}: Completed run {run_idx}/{len(triplet_sets) - 1}")
    
    print("\n" + "="*70)
    print("GENERATING PLOTS AND METRICS")
    print("="*70)
    
    # ========================================================================
    # Plot: All manual runs
    # ========================================================================
    all_data = []
    
    for model_name in MODELS_TO_TEST:
        _df = process_runs_mean(model_name, OUTPUT_BASE)
        if not _df.empty:
            _df = _df[_df['step'] >= 0]
            all_data.append(_df)
    
    if all_data:
        all_df = pd.concat(all_data, axis=0)
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=all_df, 
            x='step', 
            y='cummax FE', 
            hue='dname', 
            marker='o', 
            ms=5
        )
        plt.ylabel('Cumulative max FE (Eth)')
        plt.xlabel('Step')
        plt.title('Manual Triplet Experiments - Cumulative Max FE')
        plt.legend(title='Run', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.savefig(OUTPUT_BASE / 'manual_experiments_cummax_FE.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {OUTPUT_BASE / 'manual_experiments_cummax_FE.png'}")
        plt.close()
    
    # ========================================================================
    # Summary statistics
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY STATISTICS (MANUAL RUNS)")
    print("="*70)
    
    for model_name in MODELS_TO_TEST:
        steps_to_finish, run_details = calculate_metrics(model_name, OUTPUT_BASE)
        
        if steps_to_finish:
            sf = np.array(steps_to_finish)
            mean_steps = np.mean(sf)
            std_steps = np.std(sf)
            accel_factor = 13 / mean_steps if mean_steps > 0 else np.inf
            
            print(f"\n{model_name}:")
            print(f"  Mean steps to FE=0.245: {mean_steps:.2f} ± {std_steps:.2f}")
            print(f"  Acceleration factor: {accel_factor:.2f}x")
            print(f"\n  Individual runs:")
            for detail in run_details:
                print(f"    {detail['run']}: step={detail['step_to_threshold']}")
                print(f"      {detail['initial_triplets']}")
    
    print("\nDone!")
