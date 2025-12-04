#!/usr/bin/env python3
"""
Simple advanced transformer training script - NO HYDRA
WARNING: This model historically gets ~72% CER (much worse than baseline)
Only use this if you want to experiment with transformers
"""
import sys
sys.path.insert(0, 'src')

import os
import pickle
import time
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from neural_decoder.dataset import SpeechDataset
from neural_decoder.advanced_trainer import train_advanced_model

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_PATH = os.path.expanduser("~/competitionData/ptDecoder_ctc")
OUTPUT_DIR = os.path.expanduser("~/results/advanced_simple")

# These match your fixed advanced.yaml
args = {
    'variant': 'advanced',
    'batchSize': 16,
    'nBatch': 30000,
    'seed': 0,
    'lr': 0.005,
    'weightDecay': 0.0001,
    'nClasses': 40,
    'nInputFeatures': 256,
    'strideLen': 4,
    'kernelLen': 8,
    'gaussianSmoothWidth': 2.0,
    'modelDim': 512,
    'modelLayers': 6,
    'modelHeads': 8,
    'dropout': 0.2,
    'intermediateLayer': 3,
    'timeMaskRatio': 0.6,
    'channelDropProb': 0.3,
    'featureMaskProb': 0.1,
    'minTimeMask': 16,
    'consistencyWeight': 0.2,
    'intermediateLossWeight': 0.3,
    'testTimeLR': 0.0001,
    'enableTestTimeAdaptation': True,
    'enableOnlineAdaptation': True,
    'onlineAdaptationLR': 0.00001,
    'diphoneContext': 40,
    'transformerTimeMaskProb': 0.1,
    'relPosMaxDist': None,
    'relBiasByHead': True,
    'ffMult': 4,
    'outputDir': OUTPUT_DIR,
    'datasetPath': DATASET_PATH,
}

print("=" * 70)
print("ADVANCED TRANSFORMER MODEL TRAINING")
print("WARNING: This model typically gets ~72% CER (worse than baseline)")
print("=" * 70)
print("\nStarting training...")

train_advanced_model(args)

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
