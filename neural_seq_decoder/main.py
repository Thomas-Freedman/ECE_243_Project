#!/usr/bin/env python3
"""
Neural Decoder - Unified Training Entry Point

Usage:
    python main.py --model baseline
    python main.py --model final
    python main.py --model final2
    python main.py --model baseline --config custom_config.yaml
"""
import sys
sys.path.insert(0, 'src')

import os
import yaml
import argparse
from neural_decoder.trainer import train_model


def load_config(model_name, config_path=None):
    """
    Load configuration for specified model

    Args:
        model_name: Name of model (baseline, final, final2, advanced)
        config_path: Optional custom config path

    Returns:
        config: Configuration dictionary
    """
    if config_path is None:
        config_path = f"src/neural_decoder/conf/decoder/{model_name}.yaml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def main():
    parser = argparse.ArgumentParser(
        description='Train neural decoder models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --model baseline
  python main.py --model final --batch-size 32
  python main.py --model final2 --output ~/results/custom_run
  python main.py --model baseline --config my_config.yaml

Available models:
  baseline - 5-layer bidirectional GRU (1024 units, Adam optimizer)
  final    - 5-layer bidirectional GRU (1024 units, SGD+Nesterov)
  final2   - 7-layer Transformer (384 dim, time-masked, 83% fewer params)
  advanced - Transformer with diphone modeling (not recommended, 72% CER)
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['baseline', 'final', 'final2', 'advanced'],
        help='Model variant to train'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to custom config file (optional)'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default=os.path.expanduser("~/competitionData/ptDecoder_ctc"),
        help='Path to dataset pickle file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results (default: ~/results/{model}_run)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to train on'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size from config'
    )

    parser.add_argument(
        '--n-batch',
        type=int,
        default=None,
        help='Override number of training batches'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Override learning rate'
    )

    args = parser.parse_args()

    # Load config
    print(f"Loading configuration for model: {args.model}")
    config = load_config(args.model, args.config)

    # Override config with command line args
    if args.batch_size is not None:
        config['batchSize'] = args.batch_size
        print(f"  Overriding batch size: {args.batch_size}")

    if args.n_batch is not None:
        config['nBatch'] = args.n_batch
        print(f"  Overriding training batches: {args.n_batch}")

    if args.lr is not None:
        if 'lrStart' in config:
            config['lrStart'] = args.lr
        else:
            config['lr'] = args.lr
        print(f"  Overriding learning rate: {args.lr}")

    # Set output directory
    if args.output is None:
        output_dir = os.path.expanduser(f"~/results/{args.model}_run")
    else:
        output_dir = os.path.expanduser(args.output)

    # Verify dataset exists
    if not os.path.exists(args.dataset):
        print(f"ERROR: Dataset not found at {args.dataset}")
        print("Please format data first:")
        print("  python format_competition_data.py")
        sys.exit(1)

    # Print configuration summary
    print("\n" + "=" * 70)
    print(f"MODEL: {args.model}")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Batch size: {config['batchSize']}")
    print(f"Training batches: {config['nBatch']}")
    print("=" * 70)

    # Train model
    try:
        train_model(config, args.dataset, output_dir, args.device)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print(f"Partial results saved to: {output_dir}")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
