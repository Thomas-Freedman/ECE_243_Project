"""
Main training script with Hydra configuration.

Usage:
    python -m neural_decoder.main decoder=baseline
    python -m neural_decoder.main decoder=advanced
    python -m neural_decoder.main decoder=final
"""

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""

    print("=" * 70)
    print(f"Training with decoder variant: {cfg.decoder.variant}")
    print("=" * 70)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg.decoder))
    print("=" * 70)

    variant = cfg.decoder.variant

    if variant == "baseline":
        from .neural_decoder_trainer import trainModel
        print("\nStarting BASELINE GRU training...")
        args = OmegaConf.to_container(cfg.decoder, resolve=True)
        args["outputDir"] = cfg.outputDir
        args["datasetPath"] = cfg.datasetPath
        trainModel(args)

    elif variant == "advanced":
        from .advanced_trainer import train_advanced_model
        print("\nStarting ADVANCED Transformer training...")
        args = OmegaConf.to_container(cfg.decoder, resolve=True)
        args["outputDir"] = cfg.outputDir
        args["datasetPath"] = cfg.datasetPath
        train_advanced_model(args)

    elif variant == "final":
        from .neural_decoder_trainer import trainModel
        print("\nStarting FINAL MODEL training...")
        print("Using optimized bidirectional GRU + SGD with Nesterov momentum")
        args = OmegaConf.to_container(cfg.decoder, resolve=True)
        args["outputDir"] = cfg.outputDir
        args["datasetPath"] = cfg.datasetPath
        trainModel(args)

    else:
        raise ValueError(
            f"Unknown decoder variant: {variant}. "
            f"Choose from: baseline, advanced, final"
        )

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
