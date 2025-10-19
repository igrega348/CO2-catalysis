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
from carbondriver.loaders import load_data

NUM_RUNS = 10
MODELS = ['GP']
OUTPUT_BASE = Path('./active_learning_results')
OUTPUT_BASE.mkdir(exist_ok=True)

# Load data once
X, y, means, stds, df = load_data()
# Add triplet column
df['triplet'] = df.index // 3
df_original = df.copy()

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
    df = df_original.copy()
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
        config={'num_iter': 200, 'make_plots': False, 'normalize': True}
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
    
    # Active learning loop
    iteration = 0
    while len(chosen_triplet_ids) < df_triplet_means.shape[0]:
        # Get rows for chosen triplets
        chosen_mask = df['triplet'].isin(chosen_triplet_ids)
        chosen_df = df[chosen_mask].copy()
        chosen_df = chosen_df.drop(columns=['triplet'])
        
        # Get rows for withheld triplets
        withheld_triplet_ids = [t for t in df_triplet_means.index 
                                if t not in chosen_triplet_ids]
        withheld_mask = df['triplet'].isin(withheld_triplet_ids)
        withheld_df = df[withheld_mask].copy()
        withheld_df_triplet_col = withheld_df['triplet'].copy()
        withheld_df = withheld_df.drop(columns=['triplet'])
        
        # Evaluate acquisition function
        try:
            best_ei, best_row_idx = gde.step_within_data(chosen_df, withheld_df)
        except Exception as e:
            print(f"Error in {model_name} run {run_idx} at iteration {iteration}: {e}")
            break
        
        # Extract best triplet
        best_row_idx_int = int(best_row_idx.item()) if hasattr(best_row_idx, 'item') else int(best_row_idx)
        next_triplet_id = int(withheld_df_triplet_col.iloc[best_row_idx_int])
        
        chosen_triplet_ids.append(next_triplet_id)
        chosen_triplets_list.append(next_triplet_id)
        ei_val = float(best_ei.item()) if hasattr(best_ei, 'item') else float(best_ei)
        expected_improvements.append(ei_val)
        
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
        df_triplet_means = df_original.groupby('triplet').mean()
        
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
            try:
                run_active_learning_experiment(model, run_idx)
                if (run_idx + 1) % 10 == 0:
                    print(f"  {model}: Completed {run_idx + 1}/{NUM_RUNS} runs")
            except Exception as e:
                print(f"  {model}: Failed run {run_idx}: {e}")
    
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
            df_triplet_means = df_original.groupby('triplet').mean()
            chosen_df['cummax FE'] = df_triplet_means.loc[
                chosen_df['chosen_triplets'], 'FE (Eth)'
            ].cummax().values
            
            # Find step where cummax FE exceeds 0.245
            step_to_max = chosen_df.loc[chosen_df['cummax FE'] > 0.245, 'index'].min()
            if pd.notna(step_to_max):
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