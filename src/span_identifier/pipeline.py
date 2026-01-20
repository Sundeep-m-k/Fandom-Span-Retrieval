# src/span_identifier/pipeline.py

from src.span_identifier import paragraphs, preprocess, model


def run_span_id_pipeline(cfg: dict):
    span_cfg = cfg["span_id"]

    if not span_cfg.get("enabled", True):
        return

    # Step 1: page -> paragraph master CSV
    if span_cfg["train"].get("do_build_paragraph_master", True):
        paragraphs.build_paragraph_master_from_cfg(span_cfg)

    # Step 2: paragraph CSV -> token+BILOU JSONL
    if span_cfg["train"].get("do_preprocess", True):
        preprocess.build_token_dataset_from_cfg(span_cfg)

    # Step 3: train model
    if span_cfg["train"].get("do_train", True):
        model.train_model_from_cfg(span_cfg)

    # Step 4: evaluate model
    if span_cfg["train"].get("do_eval", True):
        model.evaluate_model_from_cfg(span_cfg)
