"""
External wrapper to run active learning experiments and generate plots.
This runs the test suite multiple times, collecting data at each step.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from carbondriver import GDEOptimizer
from carbondriver.loaders import load_gas_data

NUM_RUNS = 2
MODELS = ['GP']
OUTPUT_BASE = Path('./active_learning_results_GP')
OUTPUT_BASE.mkdir(exist_ok=True)

# Load data once
df = load_gas_data("paper/Characterization_data.xlsx")


def choose_base_inds_numpy(y: np.ndarray, num_choose: int, strategy: str = 'uniform'):
    """Choose initial triplets uniformly or skewed."""
    ind = np.argsort(y)
    N = y.shape[0]
    i = np.arange(N)
    if strategy == 'uniform':
        p = np.ones_like(i)
    else:
        raise ValueError
    p = p / p.sum()
    return np.random.choice(ind, size=num_choose, replace=False, p=p)

def run_active_learning_experiment(model_name: str, run_idx: int):
    """Run a single active learning experiment for the given model."""
    
    # Reset df for this run
    df_triplet_means = df.groupby('triplet').mean()
    
    run_dir = OUTPUT_BASE / f'{model_name}_run_{run_idx:03d}'
    run_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize optimizer
    gde = GDEOptimizer(
        model_name=model_name,
        aquisition="EI",
        quantity="FE (Eth)",
        maximize=True,
        output_dir=str(run_dir),
        config={'num_iter': 101, 'make_plots': False, 'normalize': True}
    )
    
    # Choose initial triplets
    chosen_triplet_ids = list(choose_base_inds_numpy(
        df_triplet_means['FE (Eth)'].values,
        num_choose=3,
        strategy='uniform'
    ))
    
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
        #print(best_row_idx)
        best_triplet = withheld_df.iloc[int(best_row_idx)]
        #This line ensures that we append the new triplet data to train_df, not replacing it
        train_df = df[df['triplet'] == best_triplet.name]
        withheld_df = withheld_df.drop(index=best_triplet.name)
        
        expected_improvements.append(float(best_ei))
        chosen_triplets_list.append(best_triplet.name)
        iteration += 1

    # Save results
    results_df = pd.DataFrame({
        'chosen_triplets': chosen_triplets_list,
        'expected_improvements': expected_improvements
    })
    results_df.to_csv(run_dir / 'chosen_triplets.csv')
    
    return run_dir

def process_runs_mean(model_name: str):
    """Process all runs for a given model and return aggregated DataFrame."""
    all_df = []
    
    for run_dir in OUTPUT_BASE.iterdir():
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
        
        all_df.append(chosen_df)
    
    return pd.concat(all_df, axis=0) if all_df else pd.DataFrame()

# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    # Run experiments for all models
    print(f"Running {NUM_RUNS} experiments for each model...")
    for model in MODELS:
        print(f"Running {model} experiments...")
        
        for run_idx in range(NUM_RUNS):
            print(f"\n{'='*60}")
            print(f"STARTING RUN {run_idx} ({model})")
            print(f"{'='*60}")
            run_active_learning_experiment(model, run_idx)
            print(f"  {model}: Completed run {run_idx}/{NUM_RUNS - 1}")

    
    print("\nExperiments completed! Generating plots...")
    
    # ========================================================================
    # Plot 1: 2x2 grid - one subplot per model
    # ========================================================================
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 8), sharex=True, sharey=True)
    all_data = []
    
    for i, model_name in enumerate(MODELS):
        _df = process_runs_mean(model_name)
        _df = _df[_df['step'] >= 0]
        
        sns.lineplot(
            data=_df, 
            x='step', 
            y='cummax FE', 
            hue='dname', 
            legend=False, 
            ax=ax[i // 2, i % 2]
        )
        ax[i // 2, i % 2].set_title(model_name, fontsize=12, fontweight='bold')
        ax[i // 2, i % 2].set_ylabel('Cumulative max FE (Eth)')
        ax[i // 2, i % 2].set_xlabel('Step')
        
        all_data.append(_df)
    
    fig.tight_layout()
    fig.savefig(OUTPUT_BASE / 'cummax_FE_grid.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_BASE / 'cummax_FE_grid.png'}")
    plt.close()
    
    # ========================================================================
    # Plot 2: Combined - all models on one plot
    # ========================================================================
    all_df = pd.concat(all_data, axis=0)
    
    plt.figure(figsize=(3.75, 3))
    sns.lineplot(
        data=all_df, 
        x='step', 
        y='cummax FE', 
        hue='model', 
        marker='o', 
        ms=5
    )
    plt.ylabel('Cumulative max FE (Eth)')
    plt.xlabel('Step')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_BASE / 'active_learning_combined.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_BASE / 'active_learning_combined.png'}")
    plt.close()
    
    # ========================================================================
    # Summary statistics
    # ========================================================================
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for model_name in MODELS:
        steps_to_finish = []
        for run_dir in OUTPUT_BASE.iterdir():
            if not run_dir.is_dir() or not run_dir.name.startswith(model_name):
                continue
            
            results_file = run_dir / 'chosen_triplets.csv'
            if not results_file.exists():
                continue
            
            chosen_df = pd.read_csv(results_file, index_col=0)
            df_triplet_means = df.groupby('triplet').mean()
            chosen_df['cummax FE'] = df_triplet_means.loc[
                chosen_df['chosen_triplets'], 'FE (Eth)'
            ].cummax().values
            
            # Add step column (offset by 2 due to triplet indexing)
            i0 = 2
            chosen_df['step'] = chosen_df.index - i0
            
            # Find step where cummax FE exceeds 0.245
            filtered = chosen_df[chosen_df['cummax FE'] > 0.245]
            if not filtered.empty:
                # First step where threshold was crossed
                step_to_max = filtered['step'].iloc[0]
                # Record steps to finish
                steps_to_finish.append(step_to_max)
        
        if steps_to_finish:
            sf = np.array(steps_to_finish)
            sf[sf < 0] = 0
            mean_steps = np.mean(sf)
            std_steps = np.std(sf)
            accel_factor = 13 / mean_steps if mean_steps > 0 else np.inf
            
            print(f"\n{model_name}:")
            print(f"  Mean steps to FE=0.245: {mean_steps:.2f} ± {std_steps:.2f}")
            print(f"  Acceleration factor: {accel_factor:.2f}x")
    
    print("\nDone!")
