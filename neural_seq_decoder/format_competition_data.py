#!/usr/bin/env python3
"""
Format competition data from raw .mat files into a pickle file for training.
"""

import os
import re
import pickle
import numpy as np
import scipy.io
import nltk

# Download required NLTK data
try:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('cmudict', quiet=True)
except:
    pass

from g2p_en import G2p

# Session names
sessionNames = [
    't12.2022.04.28', 't12.2022.05.26', 't12.2022.06.21', 't12.2022.07.21', 't12.2022.08.13',
    't12.2022.05.05', 't12.2022.06.02', 't12.2022.06.23', 't12.2022.07.27', 't12.2022.08.18',
    't12.2022.05.17', 't12.2022.06.07', 't12.2022.06.28', 't12.2022.07.29', 't12.2022.08.23',
    't12.2022.05.19', 't12.2022.06.14', 't12.2022.07.05', 't12.2022.08.02', 't12.2022.08.25',
    't12.2022.05.24', 't12.2022.06.16', 't12.2022.07.14', 't12.2022.08.11'
]
sessionNames.sort()

# Phoneme definitions
g2p = G2p()
PHONE_DEF = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH'
]
PHONE_DEF_SIL = PHONE_DEF + ['SIL']


def phoneToId(p):
    return PHONE_DEF_SIL.index(p)


def loadFeaturesAndNormalize(sessionPath):
    dat = scipy.io.loadmat(sessionPath)

    input_features = []
    transcriptions = []
    frame_lens = []
    n_trials = dat['sentenceText'].shape[0]

    # Collect area 6v tx1 and spikePow features
    for i in range(n_trials):
        # Get time series of TX and spike power for this trial
        # First 128 columns = area 6v only
        features = np.concatenate([dat['tx1'][0, i][:, 0:128], dat['spikePow'][0, i][:, 0:128]], axis=1)

        sentence_len = features.shape[0]
        sentence = dat['sentenceText'][i].strip()

        input_features.append(features)
        transcriptions.append(sentence)
        frame_lens.append(sentence_len)

    # Block-wise feature normalization
    blockNums = np.squeeze(dat['blockIdx'])
    blockList = np.unique(blockNums)
    blocks = []
    for b in range(len(blockList)):
        sentIdx = np.argwhere(blockNums == blockList[b])
        sentIdx = sentIdx[:, 0].astype(np.int32)
        blocks.append(sentIdx)

    for b in range(len(blocks)):
        feats = np.concatenate(input_features[blocks[b][0]:(blocks[b][-1] + 1)], axis=0)
        feats_mean = np.mean(feats, axis=0, keepdims=True)
        feats_std = np.std(feats, axis=0, keepdims=True)
        for i in blocks[b]:
            input_features[i] = (input_features[i] - feats_mean) / (feats_std + 1e-8)

    # Convert to dataset format
    session_data = {
        'inputFeatures': input_features,
        'transcriptions': transcriptions,
        'frameLens': frame_lens
    }

    return session_data


def getDataset(fileName):
    session_data = loadFeaturesAndNormalize(fileName)

    allDat = []
    trueSentences = []
    seqElements = []

    for x in range(len(session_data['inputFeatures'])):
        allDat.append(session_data['inputFeatures'][x])
        trueSentences.append(session_data['transcriptions'][x])

        thisTranscription = str(session_data['transcriptions'][x]).strip()
        thisTranscription = re.sub(r'[^a-zA-Z\- \']', '', thisTranscription)
        thisTranscription = thisTranscription.replace('--', '').lower()
        addInterWordSymbol = True

        phonemes = []
        for p in g2p(thisTranscription):
            if addInterWordSymbol and p == ' ':
                phonemes.append('SIL')
            p = re.sub(r'[0-9]', '', p)  # Remove stress
            if re.match(r'[A-Z]+', p):  # Only keep phonemes
                phonemes.append(p)

        # Add one SIL symbol at the end so there's one at the end of each word
        if addInterWordSymbol:
            phonemes.append('SIL')

        seqLen = len(phonemes)
        maxSeqLen = 500
        seqClassIDs = np.zeros([maxSeqLen]).astype(np.int32)
        seqClassIDs[0:seqLen] = [phoneToId(p) + 1 for p in phonemes]
        seqElements.append(seqClassIDs)

    newDataset = {}
    newDataset['sentenceDat'] = allDat
    newDataset['transcriptions'] = trueSentences
    newDataset['phonemes'] = seqElements

    timeSeriesLens = []
    phoneLens = []
    for x in range(len(newDataset['sentenceDat'])):
        timeSeriesLens.append(newDataset['sentenceDat'][x].shape[0])

        zeroIdx = np.argwhere(newDataset['phonemes'][x] == 0)
        phoneLens.append(zeroIdx[0, 0])

    newDataset['timeSeriesLens'] = np.array(timeSeriesLens)
    newDataset['phoneLens'] = np.array(phoneLens)
    newDataset['phonePerTime'] = newDataset['phoneLens'].astype(np.float32) / newDataset['timeSeriesLens'].astype(np.float32)
    return newDataset


def main():
    # Update this path to your competition data directory
    dataDir = os.path.expanduser('~/competitionData')
    outputFile = os.path.expanduser('~/competitionData/ptDecoder_ctc')

    print(f"Processing data from: {dataDir}")
    print(f"Output file: {outputFile}")

    trainDatasets = []
    testDatasets = []
    competitionDatasets = []

    for dayIdx in range(len(sessionNames)):
        print(f"Processing day {dayIdx}/{len(sessionNames)}: {sessionNames[dayIdx]}")

        train_path = os.path.join(dataDir, 'train', sessionNames[dayIdx] + '.mat')
        test_path = os.path.join(dataDir, 'test', sessionNames[dayIdx] + '.mat')

        print(f"  Loading train data...")
        trainDataset = getDataset(train_path)
        print(f"  Loading test data...")
        testDataset = getDataset(test_path)

        trainDatasets.append(trainDataset)
        testDatasets.append(testDataset)

        competition_path = os.path.join(dataDir, 'competitionHoldOut', sessionNames[dayIdx] + '.mat')
        if os.path.exists(competition_path):
            dataset = getDataset(competition_path)
            competitionDatasets.append(dataset)

    allDatasets = {}
    allDatasets['train'] = trainDatasets
    allDatasets['test'] = testDatasets
    allDatasets['competition'] = competitionDatasets

    print(f"\nSaving to {outputFile}...")
    with open(outputFile, 'wb') as handle:
        pickle.dump(allDatasets, handle)

    print("Done! Dataset created successfully.")
    print(f"\nDataset stats:")
    print(f"  Train days: {len(trainDatasets)}")
    print(f"  Test days: {len(testDatasets)}")
    print(f"  Competition days: {len(competitionDatasets)}")


if __name__ == "__main__":
    main()
