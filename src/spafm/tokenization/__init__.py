"""SpaFM tokenization 模块。

对外只暴露 :class:`STTokenizer`，足以把 AnnData → batch dict[str, torch.Tensor]。
"""

from spafm.tokenization.expression import bin_expression, continuous_expression
from spafm.tokenization.gene_vocab import SPECIAL_TOKENS, GeneVocab
from spafm.tokenization.spatial_encoding import rff2d, sincos2d
from spafm.tokenization.tokenizer import STTokenizer, TokenizerConfig

__all__ = [
    "SPECIAL_TOKENS",
    "GeneVocab",
    "STTokenizer",
    "TokenizerConfig",
    "bin_expression",
    "continuous_expression",
    "rff2d",
    "sincos2d",
]
