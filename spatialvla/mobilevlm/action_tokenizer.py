"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.

Codes form OpenVLA
"""

from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase
import scipy.stats as stats


class ActionTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, action_dim = 7, num_bins = 3
    ) -> None:
        self.tokenizer, self.n_tokens = tokenizer, int(num_bins ** int(action_dim))
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_tokens + 1))

        self.multiplier = np.array([num_bins ** i for i in range(action_dim)]) # N by 1
        self.detokenize_mapping = np.array([(self.to_base(i, self.num_bins) - (self.num_bins - 1) // 2) for i in range(self.n_tokens)])      
  
        self.ppf_values = []
        for b in range(1, num_bins):
            self.ppf_values.append(stats.norm.ppf(b / num_bins, loc=0, scale=1))

        self.default_token = self.discretize(np.zeros(action_dim,))

    def to_base(self, n, base):
        result = np.zeros((self.action_dim,))
        i = 0
        while n > 0:
            result[i] = n % base
            n //= base
            i += 1
        return result

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
        tokens = self.discretize(action)
        # Handle single element vs. batch
        if len(tokens.shape) == 1:
            return self.tokenizer.decode(list(tokens))
        else:
            return self.tokenizer.batch_decode(tokens.tolist())

    def discretize(self, action):
        bins = np.digitize(action, bins=self.ppf_values)
        tokens = bins @ self.multiplier
 
        return self.tokenizer.vocab_size - tokens # (1, ) for single data, or (B, 1) for batch 

    def detokenize(self, action_token_ids: np.ndarray) -> np.ndarray:
        # Just in case, If token_ids is outside of the reserved token id, just set to 0
        if self.default_token:
            mask = action_token_ids <= self.action_token_begin_idx
            action_token_ids[mask] = self.default_token

        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        return self.detokenize_mapping[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins

    
