#!/usr/bin/env python3
"""
J-Prime Golden Image Model Provisioner (Task 5)
================================================

Downloads GGUF model files defined in jarvis_prime/config/model_catalogue.yaml
into the golden-image models directory (/opt/jarvis-prime/models or
JPRIME_MODELS_DIR env override).

Usage
-----
    # Provision required models only (bake into golden image):
    python3 scripts/provision_models.py

    # Provision all models (required + optional):
    python3 scripts/provision_models.py --all

    # Override destination directory:
    JPRIME_MODELS_DIR=/tmp/models python3 scripts/provision_models.py

    # Dry-run (show what would be downloaded):
    python3 scripts/provision_models.py --dry-run

Exit codes
----------
    0  All required models present or downloaded successfully.
    1  One or more required models failed to download.
    2  Catalogue not found or invalid.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("jprime.provision")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

_CATALOGUE_PATH = (
    Path(__file__).parent.parent / "jarvis_prime" / "config" / "model_catalogue.yaml"
)


def _load_catalogue() -> Dict:
    """Load and validate model_catalogue.yaml."""
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        # Fallback: simple YAML-like parser for CI environments without PyYAML
        try:
            from jarvis_prime._compat import load_yaml as yaml_load
        except ImportError:
            raise RuntimeError(
                "PyYAML not installed — run: pip install pyyaml"
            )
    else:
        yaml_load = yaml.safe_load

    if not _CATALOGUE_PATH.exists():
        raise FileNotFoundError(f"Catalogue not found: {_CATALOGUE_PATH}")

    with _CATALOGUE_PATH.open("r") as fh:
        data = yaml_load(fh)

    if not isinstance(data, dict) or "models" not in data:
        raise ValueError("Catalogue missing 'models' section")
    return data


def _resolve_models_dir(catalogue: Dict) -> Path:
    """Determine the target models directory."""
    env_override = os.environ.get("JPRIME_MODELS_DIR")
    if env_override:
        return Path(env_override)

    provision_cfg = catalogue.get("provision", {})
    default_dest = provision_cfg.get("destination", "/opt/jarvis-prime/models")

    dest = Path(default_dest)
    if not dest.exists():
        # Fall back to dev destination
        dev_dest = provision_cfg.get("dev_destination", "~/.jarvis/prime/models")
        dest = Path(dev_dest).expanduser()
        logger.info("Golden-image dir not found; using dev dir: %s", dest)

    return dest


def _is_present(models_dir: Path, gguf_file: str) -> bool:
    return (models_dir / gguf_file).exists()


def _download_model(
    models_dir: Path,
    model_name: str,
    entry: Dict,
    dry_run: bool = False,
) -> bool:
    """Download a single GGUF via huggingface_hub.

    Returns True on success, False on failure.
    """
    gguf_file: str = entry["gguf_file"]
    dest = models_dir / gguf_file

    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info("[%s] Already present: %s (%.0f MB)", model_name, gguf_file, size_mb)
        return True

    hf_repo: str = entry.get("hf_repo", "")
    hf_filename: str = entry.get("hf_filename", gguf_file)

    if not hf_repo:
        logger.warning("[%s] No hf_repo defined — cannot download", model_name)
        return False

    if dry_run:
        logger.info(
            "[DRY-RUN] Would download %s/%s → %s", hf_repo, hf_filename, dest
        )
        return True

    logger.info("[%s] Downloading %s/%s …", model_name, hf_repo, hf_filename)
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import]

        models_dir.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=hf_repo,
            filename=hf_filename,
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download may place files in subdirs — move to flat models_dir
        downloaded_path = Path(downloaded)
        if downloaded_path != dest and downloaded_path.exists():
            downloaded_path.rename(dest)

        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info("[%s] Downloaded: %s (%.0f MB)", model_name, dest.name, size_mb)
        return True

    except ImportError:
        logger.error(
            "[%s] huggingface_hub not installed — run: pip install huggingface_hub",
            model_name,
        )
        return False
    except Exception as exc:
        logger.error("[%s] Download failed: %s", model_name, exc)
        return False


def provision(
    download_all: bool = False,
    dry_run: bool = False,
    model_filter: Optional[List[str]] = None,
) -> int:
    """Provision models.

    Returns 0 on full success, 1 on partial failure.
    """
    try:
        catalogue = _load_catalogue()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("Cannot load catalogue: %s", exc)
        return 2

    models_dir = _resolve_models_dir(catalogue)
    provision_cfg = catalogue.get("provision", {})
    all_models: Dict[str, Dict] = catalogue.get("models", {})

    # Determine which models to download
    required: List[str] = provision_cfg.get("required", [])
    optional: List[str] = provision_cfg.get("optional", []) if download_all else []
    targets = list(dict.fromkeys(required + optional))  # dedup, preserve order

    if model_filter:
        targets = [m for m in targets if m in model_filter]

    if not targets:
        logger.warning("No models selected for provisioning")
        return 0

    logger.info("Target directory: %s", models_dir)
    logger.info("Models to provision: %s", targets)

    failures: List[str] = []
    for model_name in targets:
        entry = all_models.get(model_name)
        if entry is None:
            logger.error("[%s] Not found in catalogue — skipping", model_name)
            if model_name in required:
                failures.append(model_name)
            continue

        ok = _download_model(models_dir, model_name, entry, dry_run=dry_run)
        if not ok and model_name in required:
            failures.append(model_name)

    if failures:
        logger.error("Required models failed to provision: %s", failures)
        return 1

    logger.info("Provisioning complete. Models directory: %s", models_dir)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provision J-Prime GGUF models for golden image baking."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download optional models in addition to required ones.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help="Download only these specific model names.",
    )
    args = parser.parse_args()
    sys.exit(provision(
        download_all=args.all,
        dry_run=args.dry_run,
        model_filter=args.models,
    ))


if __name__ == "__main__":
    main()
