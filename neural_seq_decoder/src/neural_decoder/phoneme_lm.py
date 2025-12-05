"""
Phoneme Language Model (LM) Integration for Neural Decoders
Uses KenLM for phoneme sequence validation and beam search decoding
"""
import os
import numpy as np
import torch
from typing import List, Optional

# Try to import kenlm
try:
    import kenlm
    LM_AVAILABLE = True
except ImportError:
    kenlm = None
    LM_AVAILABLE = False
    print("WARNING: kenlm not installed. Install with: pip install https://github.com/kpu/kenlm/archive/master.zip")


# Phoneme definitions (CTC blank=0, phonemes 1-40)
PHONE_DEF = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH', 'SIL'
]

# Create phoneme mappings (0=blank, 1-40=phonemes)
def create_phoneme_map():
    """Create mapping from phoneme IDs to phoneme strings"""
    phoneme_map = {0: '<blank>'}  # CTC blank
    for i, phone in enumerate(PHONE_DEF, start=1):
        phoneme_map[i] = phone
    return phoneme_map


class PhonemeLM:
    """
    Phoneme Language Model wrapper for KenLM
    Provides scoring and beam search decoding
    """
    def __init__(self, lm_path: str, phoneme_map: dict = None):
        if not LM_AVAILABLE:
            raise RuntimeError(
                "kenlm not available. Install with: "
                "pip install https://github.com/kpu/kenlm/archive/master.zip"
            )
        if not os.path.exists(lm_path):
            raise FileNotFoundError(f"LM file not found: {lm_path}")

        self.model = kenlm.Model(lm_path)
        self.phoneme_map = phoneme_map or create_phoneme_map()
        self._score_cache = {}

    def id_sequence_to_tokens(self, id_seq):
        """Convert phoneme IDs to token strings"""
        tokens = []
        for i in id_seq:
            t = self.phoneme_map.get(int(i), None)
            if t is None or t == '<blank>':
                continue  # Skip blank and unknown
            tokens.append(t)
        return tokens

    def score(self, id_seq):
        """
        Score a phoneme sequence using the LM
        Returns log probability (natural log)
        """
        # Use tuple for hashable cache key
        key = tuple(id_seq) if not isinstance(id_seq, tuple) else id_seq
        if key in self._score_cache:
            return self._score_cache[key]

        tokens = self.id_sequence_to_tokens(id_seq)
        if len(tokens) == 0:
            return 0.0

        # KenLM expects space-separated string
        text = " ".join(tokens)
        s = self.model.score(text, bos=False, eos=False)
        self._score_cache[key] = float(s)
        return float(s)

    def clear_cache(self):
        """Clear the LM score cache"""
        self._score_cache.clear()


def beam_search_decode(
    log_probs: torch.Tensor,
    lm: Optional[PhonemeLM] = None,
    lm_weight: float = 0.8,
    beam_width: int = 10,
    blank_id: int = 0,
    topk_acoustic: int = 5
) -> List[int]:
    """
    Beam search decoding with optional LM fusion

    Args:
        log_probs: [T, V] log-probabilities from acoustic model
        lm: PhonemeLM instance (optional)
        lm_weight: Weight for LM score (0.0 = pure acoustic, 1.0 = pure LM)
        beam_width: Number of beams to keep
        blank_id: CTC blank token ID (usually 0)
        topk_acoustic: Number of top acoustic candidates per timestep

    Returns:
        List of phoneme IDs (without blanks, without consecutive duplicates)
    """
    # Convert to numpy if tensor
    if isinstance(log_probs, torch.Tensor):
        lp = log_probs.detach().cpu().numpy()
    else:
        lp = np.array(log_probs)

    T, V = lp.shape

    # Each beam: (sequence, acoustic_score, lm_score)
    beams = [([], 0.0, 0.0)]

    for t in range(T):
        step = lp[t]  # [V]

        # Get top-k acoustic candidates to limit branching
        topk_idx = np.argsort(step)[-topk_acoustic:][::-1]  # Descending

        new_beams = {}
        for seq, a_score, l_score in beams:
            for idx in topk_idx:
                token_logp = float(step[idx])

                if idx == blank_id:
                    # Blank: keep sequence unchanged
                    new_seq = tuple(seq)
                    new_a = a_score + token_logp
                    new_l = l_score  # LM score unchanged
                else:
                    # Append non-blank token
                    new_seq = tuple(list(seq) + [int(idx)])
                    new_a = a_score + token_logp

                    # Compute LM score if available
                    if lm is not None:
                        new_l = lm.score(new_seq)
                    else:
                        new_l = 0.0

                # Combined score
                combined = new_a + (lm_weight * new_l)

                # Keep best score for each unique sequence
                if new_seq not in new_beams or combined > new_beams[new_seq][0]:
                    new_beams[new_seq] = (combined, new_a, new_l)

        # Prune to beam_width
        sorted_beams = sorted(
            new_beams.items(),
            key=lambda x: x[1][0],  # Sort by combined score
            reverse=True
        )[:beam_width]

        beams = [(list(k), v[1], v[2]) for k, v in sorted_beams]

    # Select best beam
    best = max(beams, key=lambda b: b[1] + lm_weight * b[2])
    decoded = best[0]

    # Collapse repeated consecutive tokens (CTC-style)
    collapsed = []
    prev = None
    for tok in decoded:
        if tok == prev:
            continue
        if tok != blank_id:
            collapsed.append(tok)
        prev = tok

    return collapsed


def greedy_decode_with_lm(
    log_probs: torch.Tensor,
    lm: Optional[PhonemeLM] = None,
    lm_weight: float = 0.3,
    blank_id: int = 0
) -> List[int]:
    """
    Greedy decoding with optional LM rescoring
    Faster than beam search but less accurate

    Args:
        log_probs: [T, V] log-probabilities
        lm: PhonemeLM instance (optional)
        lm_weight: Weight for LM score
        blank_id: CTC blank token ID

    Returns:
        List of phoneme IDs
    """
    if isinstance(log_probs, torch.Tensor):
        lp = log_probs.detach().cpu().numpy()
    else:
        lp = np.array(log_probs)

    T, V = lp.shape

    # Greedy selection at each timestep
    decoded_raw = []
    for t in range(T):
        idx = int(np.argmax(lp[t]))
        decoded_raw.append(idx)

    # Collapse CTC (remove blanks and consecutive duplicates)
    collapsed = []
    prev = None
    for tok in decoded_raw:
        if tok == prev:
            continue
        if tok != blank_id:
            collapsed.append(tok)
        prev = tok

    # Optional LM rescoring (doesn't change sequence, just for debugging)
    if lm is not None and len(collapsed) > 0:
        lm_score = lm.score(collapsed)
        # In greedy mode, we don't change the sequence, just report the score
        # To actually use LM, use beam_search_decode instead

    return collapsed
