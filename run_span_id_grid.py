# scripts/run_span_id_grid.py

import subprocess
import yaml
from pathlib import Path

def run_experiment(level, normalize_punct, domain="money-heist"):
    """Run one experiment configuration."""
    
    config_path = Path("configs/span_id.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    cfg["span_id"]["level"] = level
    cfg["span_id"]["normalize_punctuation"] = normalize_punct
    
    exp_name = f"{level}_{'punc' if normalize_punct else 'no_punc'}"
    cfg["span_id"]["token_dataset_dir"] = f"data/processed/{domain}/span_id_{exp_name}"    
    temp_cfg_path = Path(f"configs/temp_{exp_name}.yaml")
    with open(temp_cfg_path, "w") as f:
        yaml.dump(cfg, f)
    
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"{'='*60}\n")
    
    subprocess.run(["python", "main_span_id.py", "--config", str(temp_cfg_path)])
    temp_cfg_path.unlink()

if __name__ == "__main__":
    for level in ["paragraph", "article"]:
        for normalize_punct in [False, True]:
            run_experiment(level, normalize_punct)
    
    print("\n" + "="*60)
    print("All experiments complete!")
    print(f"Results saved to: results/span_identification/all_experiments.csv")
    print("="*60)
