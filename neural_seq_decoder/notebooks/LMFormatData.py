from g2p_en import G2p
import re
import os
from pathlib import Path

g2p = G2p()

def concatenate_files(input_files, output_file):
    with open(output_file, 'w') as outfile:
        for fname in input_files:
            with open(fname, 'r') as infile:
                outfile.write(infile.read())

def get_txt_files(input_dir):
    path = Path(input_dir)
    return [str(f) for f in path.rglob(f"*.txt")]

def generate_dataset(input_files, output_file, addInterWordSymbol=True):
    with open(output_file, "w") as out:
        for in_file in input_files:
            with open(in_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(" ", 1)
                    if len(parts) < 2:
                        continue

                    utt_id, text = parts[0], parts[1]

                    text = re.sub(r'[^a-zA-Z\- \']', '', text)
                    text = text.replace('--', '').lower()
                    addInterWordSymbol = True

                    phonemes = []
                    for p in g2p(text):
                        if addInterWordSymbol and p==' ':
                            phonemes.append('SIL')
                        p = re.sub(r'[0-9]', '', p)  # Remove stress
                        if re.match(r'[A-Z]+', p):  # Only keep phonemes
                            phonemes.append(p)

                    #add one SIL symbol at the end so there's one at the end of each word
                    if addInterWordSymbol:
                        phonemes.append('SIL')

                    out.write(" ".join(phonemes) + "\n")

# USAGE #

# input_dir = "/Users/tnfreedman/Downloads/LibriSpeech3/train-clean-360"
# output_file = "phoneme_lm_train3.txt"

# input_files = get_txt_files(input_dir)
# print(len(input_files))

# generate_dataset(input_files, output_file)    



input1 = 'neural_seq_decoder/LM_DATA/phoneme_lm_train1.txt'
input2 = 'neural_seq_decoder/LM_DATA/phoneme_lm_train2.txt'
input3 = 'neural_seq_decoder/LM_DATA/phoneme_lm_train3.txt'

inputs = [input1, input2, input3]

output = 'neural_seq_decoder/LM_DATA/phoneme_lm_train_FULL.txt'

concatenate_files(inputs, output)

