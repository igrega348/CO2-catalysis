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
from typing import Tuple, Optional, Literal
import torch
import yaml
import sys

def choose_base_inds_numpy(y: np.ndarray, num_choose: int, how: Literal['max','min'] = 'max', strategy: Literal['uniform','skewed'] = 'uniform', seed: Optional[int] = None):
    ind = np.argsort(y)
    N = y.shape[0]
    i = np.arange(N)
    if strategy=='skewed':
        if how=='max':
            p = (i - i.max())**2
        elif how=='min':
            p = i**2
    elif strategy=='uniform':
        p = np.ones_like(i)
    else: 
        raise ValueError
    p = p/p.sum()
    rng = np.random.default_rng(seed)
    return rng.choice(ind, size=num_choose, replace=False, p=p)

def run_active_learning_experiment(model_name: str, run_idx: int):
    """Run a single active learning experiment for the given model."""
    
    print(f"  [Step 1/3] Preparing data for {model_name} run {run_idx}...")
    # Reset df for this run
    df_triplet_means = df.groupby('triplet').mean()
    
    run_dir = OUTPUT_BASE / f'{model_name}_run_{run_idx:03d}'
    run_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"  [Step 2/3] Initializing {model_name} optimizer...")
    # Initialize optimizer
    gde = GDEOptimizer(
        model_name=model_name,
        aquisition="EI",
        quantity="FE (Eth)",
        maximize=True,
        output_dir=str(run_dir),
        config=config,
    )
    
    # Choose initial triplets
    print(f"  [Step 3/3] Selecting initial triplets...")
    chosen_triplets_ids = choose_base_inds_numpy(
        df_triplet_means[config["property_name"]].values,
        num_choose=3,
        strategy='uniform',
        seed=run_idx
    ).tolist()

    bests = df_triplet_means.loc[chosen_triplets_ids][config["property_name"]].cummax().tolist()
    print(f"    Starting triplets: {chosen_triplets_ids}")
    print(f"    Starting best FE values: {[f'{b:.4f}' for b in bests]}")

    # Track results
    expected_improvements = [None] * len(chosen_triplets_ids)
    nll_values = [None] * len(chosen_triplets_ids)  # NLL from training
    loss_values = [None] * len(chosen_triplets_ids)  # Loss (MSE proxy) from training
    # Get rows for chosen triplets
    train_df = df[df['triplet'].isin(chosen_triplets_ids)].copy()
    # Get rows for withheld triplets
    withheld_df = df_triplet_means[~df_triplet_means.index.isin(chosen_triplets_ids)].copy()

    # Active learning loop
    iteration = 0
    
    print(f"\n  Starting active learning loop with {len(withheld_df)} candidates...")
    while len(withheld_df) > 0:
        print(f"  Run {run_idx}, Iteration {iteration}: Evaluating acquisition function...")
        # Evaluate acquisition function
        best_ei, best_row_idx, metrics = gde.step_within_data(train_df, withheld_df, return_metrics=True)
        best_triplet = withheld_df.iloc[int(best_row_idx)]
        #This line ensures that we append the new triplet data to train_df, not replacing it
        train_df = df[df['triplet'] == best_triplet.name]
        withheld_df = withheld_df.drop(index=best_triplet.name)
        
        expected_improvements.append(float(best_ei))
        nll_values.append(metrics.get('nll', None))
        loss_values.append(metrics.get('loss', None))
        chosen_triplets_ids.append(int(best_triplet.name))

        current_best = df_triplet_means.loc[chosen_triplets_ids][config["property_name"]].max().item()
        bests.append(current_best)
        print(f"    ✓ Selected triplet {int(best_triplet.name)}, Best FE: {current_best:.4f}, EI: {best_ei:.6f}, Remaining candidates: {len(withheld_df)}")
        
        iteration += 1

        if best_triplet.name == df_triplet_means[config["property_name"]].idxmax():
            print(f"    ✓ Found global optimum! Stopping early.")
            break

    print(f"  ✓ Run {run_idx} completed in {iteration} iterations.\n")
    # Save results
    results_df = pd.DataFrame({
        'chosen_triplets': chosen_triplets_ids,
        'expected_improvements': expected_improvements,
        'nll': nll_values,
        'loss': loss_values
    })
    results_df.to_csv(run_dir / 'chosen_triplets.csv')
    print(f"  Results saved to {run_dir / 'chosen_triplets.csv'}")
    
    return run_dir

def process_runs_mean(model_name: str):
    """Process all runs for a given model and return aggregated DataFrame."""
    print(f"  Processing results for {model_name}...")
    all_df = []
    run_count = 0
    
    for run_dir in OUTPUT_BASE.glob(model_name + '_run_*/'):
        if not run_dir.is_dir():
            continue
        
        results_file = run_dir / 'chosen_triplets.csv'
        if not results_file.exists():
            continue
        
        # Read the results
        chosen_df = pd.read_csv(results_file, index_col=0)
        df_triplet_means = df.groupby('triplet').mean()
        
        # Calculate cummax FE for this run
        chosen_df['cummax FE'] = df_triplet_means.loc[
            chosen_df['chosen_triplets'], config["property_name"]
        ].cummax().values
        
        # Add step column (offset by 2 to match old convention)
        i0 = 2
        chosen_df['step'] = chosen_df.index - i0
        chosen_df['dname'] = run_dir.stem
        chosen_df['model'] = model_name
        
        all_df.append(chosen_df)
        run_count += 1
    
    print(f"    ✓ Loaded {run_count} runs for {model_name}")
    return pd.concat(all_df, axis=0) if all_df else pd.DataFrame()

# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':

    print("="*70)
    print("ACTIVE LEARNING EXPERIMENT SUITE")
    print("="*70)
    
    print("\n[1/6] Loading configuration...")
    with open(sys.argv[1]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    OUTPUT_BASE = Path(config["run_name"])
    print(f"  Output directory: {OUTPUT_BASE}")
    OUTPUT_BASE.mkdir(exist_ok=True)

    # Load data once
    print("[2/6] Loading experimental data...")
    df = load_gas_data("paper/Characterization_data.xlsx")
    print(f"  ✓ Loaded {len(df)} data points")
    
    if not config["use_existing_results"]:
        
        print("\n[3/6] Running new experiments...")
        torch.manual_seed(config["torch_seed"])
        print(f"  Set torch seed to {config['torch_seed']}")
        
        # Run experiments for all models
        total_runs = len(config["models"]) * len(list(config.get("runs", range(config["num_runs"]))))
        run_num = 0
        
        for model in config["models"]:
            print(f"\n  → Starting experiments for {model.upper()}...")
            
            for run_idx in config.get("runs", range(config["num_runs"])):
                run_num += 1
                print(f"\n{'='*70}")
                print(f"RUN {run_num}/{total_runs}: {model.upper()} (Seed {run_idx})")
                print(f"{'='*70}")
                run_active_learning_experiment(model, run_idx)
                print(f"✓ Completed {model} run {run_idx+1}/{config['num_runs']}")
                
        print("\n" + "="*70)
        print("✓ All experiments completed!")
        print("="*70)
        print("\nGenerating plots...")
    else:
        print("\n[3/6] Using existing results (use_existing_results=True)")
    
    # ========================================================================
    # Plot 1: 2x2 grid - one subplot per model
    # ========================================================================
    print("\n[4/6] Generating 2x2 grid plot...")
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 8), sharex=True, sharey=True)
    all_data = []
    
    for i, model_name in enumerate(config["models"]):
        print(f"  Processing subplot for {model_name}...")
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
    print(f"  ✓ Saved: {OUTPUT_BASE / 'cummax_FE_grid.png'}")
    plt.close()
    
    # ========================================================================
    # Plot 2: Combined - all models on one plot
    # ========================================================================
    print("\n[5/6] Generating combined model comparison plot...")
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
    print(f"  ✓ Saved: {OUTPUT_BASE / 'active_learning_combined.png'}")
    plt.close()
    
    # Reset index to avoid duplicate label issues in seaborn
    all_df = all_df.reset_index(drop=True)
    
    # ========================================================================
    # Plot 3: NLL over steps (training quality metric)
    # ========================================================================
    if 'nll' in all_df.columns:
        print("  Generating NLL vs step plot...")
        plt.figure(figsize=(5, 4))
        sns.lineplot(
            data=all_df[all_df['step'] >= 0], 
            x='step', 
            y='nll', 
            hue='model',
            errorbar='sd'
        )
        plt.ylabel('NLL (Negative Log-Likelihood)')
        plt.xlabel('Step')
        plt.title('Model Uncertainty (NLL) vs Active Learning Step')
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(OUTPUT_BASE / 'nll_over_steps.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {OUTPUT_BASE / 'nll_over_steps.png'}")
        plt.close()
    
    # ========================================================================
    # Plot 4: Loss (MSE) over steps (training fit quality)
    # ========================================================================
    if 'loss' in all_df.columns:
        print("  Generating Loss vs step plot...")
        plt.figure(figsize=(5, 4))
        sns.lineplot(
            data=all_df[all_df['step'] >= 0], 
            x='step', 
            y='loss', 
            hue='model',
            errorbar='sd'
        )
        plt.ylabel('Loss (Neg. Marginal Log-Likelihood for GP)')
        plt.xlabel('Step')
        plt.title('Training Loss vs Active Learning Step')
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(OUTPUT_BASE / 'loss_over_steps.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {OUTPUT_BASE / 'loss_over_steps.png'}")
        plt.close()
    
    # ========================================================================
    # Summary statistics
    # ========================================================================
    print("\n[6/6] Computing summary statistics...")
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    for model_name in config["models"]:
        print(f"\n{model_name.upper()}:")
        steps_to_finish = []
        final_nlls = []
        final_losses = []
        run_count = 0
        
        for run_dir in OUTPUT_BASE.glob(model_name + '_run_*/'):
            if not run_dir.is_dir() or not run_dir.name.startswith(model_name):
                continue
            
            results_file = run_dir / 'chosen_triplets.csv'
            if not results_file.exists():
                continue
            
            run_count += 1
            chosen_df = pd.read_csv(results_file, index_col=0)
            df_triplet_means = df.groupby('triplet').mean()
            chosen_df['cummax FE'] = df_triplet_means.loc[
                chosen_df['chosen_triplets'], config["property_name"]
            ].cummax().values
            
            # Add step column (offset by 2 due to triplet indexing)
            i0 = 2
            chosen_df['step'] = chosen_df.index - i0
            
            # Collect final NLL and loss values (last non-null).
            # Simplified: try to append the last non-NaN value for each column,
            # skip silently if the column is missing or contains only NaNs.
            for col, storage in (('nll', final_nlls), ('loss', final_losses)):
                try:
                    storage.append(chosen_df[col].dropna().iat[-1])
                except Exception:
                    # missing column or all-NaN -> skip
                    pass
            
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
            
            print(f"  Runs analyzed: {run_count}")
            print(f"  Mean steps to FE=0.245: {mean_steps:.2f} ± {std_steps:.2f}")
            if final_losses:
                print(f"  Final MSE (loss): {np.mean(final_losses):.4f} ± {np.std(final_losses):.4f}")
            if final_nlls:
                print(f"  Final NLL: {np.mean(final_nlls):.4f} ± {np.std(final_nlls):.4f}")
            print(f"  Acceleration factor: {accel_factor:.2f}x")
        else:
            print(f"  No data available for {model_name}")
    
    print("\n" + "="*70)
    print("✓ Analysis complete!")
    print("="*70 + "\n")
