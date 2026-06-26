import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

def extract_data_from_path(base_path, prefix):
    data_list = []
    root_dir = Path(base_path)
    csv_files = list(root_dir.glob(f"{prefix}*/**/*.csv"))
    
    # Pattern to extract (Epoch)-(Step)-(Loss) from folder name
    # e.g.: 35-29052-4.3491 -> epoch: 35, step: 29052, loss: 4.3491
    pattern = re.compile(r"(\d+)-(\d+)-([\d.]+)")

    for csv_path in csv_files:
        try:
            folder_name = csv_path.parts[csv_path.parts.index(root_dir.name) + 1] if root_dir.name in csv_path.parts else csv_path.parts[-3]
            ckpt_id = folder_name.replace(f"{prefix}", "")
            
            # Regex matching
            match = pattern.search(ckpt_id)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                val_loss = float(match.group(3))
                
                df = pd.read_csv(csv_path)
                if 'score' in df.columns:
                    final_score = df['score'].iloc[-1]
                    
                    data_list.append({
                        'ckpt_id': ckpt_id,
                        'epoch': epoch,
                        'step': step,
                        'val_loss': val_loss,
                        'score': final_score
                    })
            
        except Exception as e:
            print(f"[ERROR] Could not process {csv_path}: {e}")
            
    return pd.DataFrame(data_list)

def plot_visualizations(df, prefix):
    if df.empty:
        print("No data to visualize.")
        return

    # Sort data by epoch
    df = df.sort_values('epoch')

    # Style configuration
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # 1. Score by Epoch (Line Plot)
    sns.lineplot(ax=axes[0], data=df, x='epoch', y='score', marker='o', color='royalblue', linewidth=2)
    axes[0].set_title('Score Trend by Epoch', fontsize=15, pad=15)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)

    # 2. Score by Validation Loss (Scatter Plot)
    sns.scatterplot(ax=axes[1], data=df, x='val_loss', y='score', hue='epoch', size='epoch', 
                    palette='viridis', sizes=(50, 200), alpha=0.8)
    axes[1].set_title('Score vs Validation Loss (from Path)', fontsize=15, pad=15)
    axes[1].set_xlabel('Validation Loss (extracted)', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].legend(title='Epoch', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(f"{prefix}_epdms.png")
    plt.show()

if __name__ == "__main__":
    EXP_ROOT = "/workspace/PriorEye/exp"
    prefix = "gtrs_dense_baseline_map_vector_navhard_two_stage"

    # 1. Extract data
    results_df = extract_data_from_path(EXP_ROOT, prefix)
    
    if not results_df.empty:
        # 2. Print results (keep existing text output)
        print("\n" + "="*50)
        print(f"{'Checkpoint ID':<40} | {'Score':<7}")
        print("-"*50)
        sorted_df = results_df.sort_values(by='score', ascending=False)
        for _, row in sorted_df.iterrows():
            print(f"{row['ckpt_id']:<40} | {row['score']:.4f}")
            
        # 3. Visualization
        plot_visualizations(results_df, prefix)
    else:
        print("No data found. Please check the path and prefix.")