import torch
import torch.nn.functional as F
from typing import List, Tuple
import kenlm

PHONE_DEF = [
    'AA','AE','AH','AO','AW',
    'AY','B','CH','D','DH',
    'EH','ER','EY','F','G',
    'HH','IH','IY','JH','K',
    'L','M','N','NG','OW',
    'OY','P','R','S','SH',
    'T','TH','UH','UW','V',
    'W','Y','Z','ZH'
]

PHONE_DEF_SIL = PHONE_DEF + ['SIL']
index2phone = {i:p for i,p in enumerate(PHONE_DEF_SIL)}
phone2index = {p:i for i,p in enumerate(PHONE_DEF_SIL)}

# ---------------------------
# LANGUAGE MODEL
# ---------------------------
class PhonemeLM:
    def __init__(self, lm_path: str):
        self.model = kenlm.Model(lm_path)

    def score(self, tokens: List[str]) -> float:
        line = " ".join(tokens)
        return self.model.score(line, bos=False, eos=False)

# ---------------------------
# BEAM SEARCH
# ---------------------------
def lm_beam_search(
    logits: torch.Tensor,      
    lm: PhonemeLM,
    beam_size: int = 10,
    lm_weight: float = 0.4,
    length_penalty: float = 0.0
):

    T, V = logits.size()
    log_probs = F.log_softmax(logits, dim=-1)

    beam = [([], 0.0, 0.0)]

    for t in range(T):
        new_beam = []
        lp_t = log_probs[t]

        for tokens, a_score, lm_score in beam:
            for v in range(V):
                new_tok = tokens + [index2phone[v]]
                new_a = a_score + lp_t[v].item()
                new_lm = lm.score(new_tok)

                fused = new_a + lm_weight * new_lm

                new_beam.append((new_tok, new_a, new_lm, fused))

        new_beam = sorted(new_beam, key=lambda x: x[3], reverse=True)[:beam_size]
        beam = [(toks, a, l) for (toks, a, l, _) in new_beam]

    beam = sorted(beam, key=lambda x: x[1] + lm_weight * x[2], reverse=True)

    return beam[0][0]   







from phoneme_lm_decoder import *
import torch

# --------------------------
# Load your acoustic model
# --------------------------
model = YourModelClass(...)
model.load_state_dict(torch.load("your_model.pt", map_location="cpu"))
model.eval()

# --------------------------
# Load KenLM Model
# --------------------------
lm = PhonemeLM("phoneme_lm.bin")   # or .arpa

# --------------------------
# Run inference
# --------------------------
with torch.no_grad():
    logits = model(x)    # logits: [T, vocab]

decoded = lm_beam_search(
    logits.squeeze(0),
    lm,
    beam_size=20,
    lm_weight=0.4,
)

print("Decoded phonemes:")
print(decoded)
