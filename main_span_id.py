# main_span_id.py
import yaml
from copy import deepcopy
from src.span_identifier.pipeline import run_span_id_pipeline

def interpolate_span_cfg(cfg):
    """Replace ${...} placeholders in span_id config using simple string formatting."""
    span_cfg = deepcopy(cfg["span_id"])
    domain = span_cfg["domain"]

    # First-level paths
    span_cfg["text_dir"] = span_cfg["text_dir"].replace("${domain}", domain)
    span_cfg["spans_csv_path"] = span_cfg["spans_csv_path"].replace("${domain}", domain)
    span_cfg["paragraph_master_csv"] = span_cfg["paragraph_master_csv"].replace("${domain}", domain)
    span_cfg["token_dataset_dir"] = span_cfg["token_dataset_dir"].replace("${domain}", domain)
    span_cfg["log_dir"] = span_cfg["log_dir"].replace("${domain}", domain)

    # Derived from token_dataset_dir
    tdir = span_cfg["token_dataset_dir"]
    span_cfg["train_file"] = span_cfg["train_file"].replace("${token_dataset_dir}", tdir)
    span_cfg["dev_file"] = span_cfg["dev_file"].replace("${token_dataset_dir}", tdir)
    span_cfg["test_file"] = span_cfg["test_file"].replace("${token_dataset_dir}", tdir)

    # Logging name
    if "logging" in span_cfg:
        span_cfg["logging"]["wandb_run_name"] = span_cfg["logging"]["wandb_run_name"].replace("${domain}", domain)

    # Put back
    cfg["span_id"] = span_cfg
    return cfg

def main():
    with open("configs/span_id.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg = interpolate_span_cfg(cfg)
    run_span_id_pipeline(cfg)

if __name__ == "__main__":
    main()
