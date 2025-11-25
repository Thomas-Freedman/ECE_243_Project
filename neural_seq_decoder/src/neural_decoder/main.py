"""
Main training script with Hydra configuration.

Usage:
    python -m neural_decoder.main decoder=baseline
    python -m neural_decoder.main decoder=advanced
    python -m neural_decoder.main decoder=hybrid
    python -m neural_decoder.main decoder=poyo
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
        trainModel(OmegaConf.to_container(cfg.decoder, resolve=True))

    elif variant == "advanced":
        from .advanced_trainer import train_advanced_model
        print("\nStarting ADVANCED Transformer training...")
        train_advanced_model(OmegaConf.to_container(cfg.decoder, resolve=True))

    elif variant == "hybrid":
        from .advanced_trainer import train_hybrid_model
        print("\nStarting HYBRID (Transformer+SSM) training...")
        train_hybrid_model(OmegaConf.to_container(cfg.decoder, resolve=True))

    elif variant == "poyo":
        from .advanced_trainer import train_poyo_model
        print("\nStarting POYO (spike tokenization) training...")
        train_poyo_model(OmegaConf.to_container(cfg.decoder, resolve=True))

    else:
        raise ValueError(
            f"Unknown decoder variant: {variant}. "
            f"Choose from: baseline, advanced, hybrid, poyo"
        )

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
