"""
Sub-token Chess Tokenizer for the Chess Challenge.

This tokenizer decomposes each move into a small set of structural tokens:
- Color
- Piece
- From square
- To square
- Promotion
- Suffix (capture/check/mate/castling)
- Move separator token (<SP>) which decodes to a whitespace " "

It is designed to work with the provided evaluate.py which generates tokens
until it encounters a separator token (whitespace / EOS).
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from transformers import PreTrainedTokenizer


_MOVE_RE = re.compile(
    r"^"
    r"(?P<color>[WB])"
    r"(?P<piece>[PNBRQK])"
    r"(?P<from_sq>[a-h][1-8])"
    r"(?P<to_sq>[a-h][1-8])"
    r"(?P<promo>=[NBRQ])?"
    r"(?P<suffix>\([^)]*\))?"
    r"$"
)


class ChessTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"vocab_file": "vocab.json"}

    # Special tokens
    PAD_TOKEN = "[PAD]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"

    # Move separator (MUST decode to whitespace so evaluate.py stops on it)
    SP_TOKEN = "<SP>"

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        vocab: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        # Define special tokens
        self._pad_token = self.PAD_TOKEN
        self._bos_token = self.BOS_TOKEN
        self._eos_token = self.EOS_TOKEN
        self._unk_token = self.UNK_TOKEN

        # Avoid duplicates passed via kwargs
        kwargs.pop("pad_token", None)
        kwargs.pop("bos_token", None)
        kwargs.pop("eos_token", None)
        kwargs.pop("unk_token", None)

        # IMPORTANT for sub-token moves: we want to keep the most recent tokens
        # when sequences are too long (evaluation will exceed n_ctx quickly).
        # This makes truncation keep the RIGHT side (latest moves).
        self.truncation_side = "left"
        self.padding_side = "right"

        if vocab is not None:
            self._vocab = vocab
        elif vocab_file is not None and os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                self._vocab = json.load(f)
        else:
            self._vocab = self._create_fixed_vocab()

        self._ids_to_tokens = {v: k for k, v in self._vocab.items()}

        super().__init__(
            pad_token=self._pad_token,
            bos_token=self._bos_token,
            eos_token=self._eos_token,
            unk_token=self._unk_token,
            **kwargs,
        )

    # ---------- Vocab ----------
    @classmethod
    def _all_squares(cls) -> List[str]:
        files = "abcdefgh"
        ranks = "12345678"
        return [f"{f}{r}" for r in ranks for f in files]

    @classmethod
    def _create_fixed_vocab(cls) -> Dict[str, int]:
        special = [cls.PAD_TOKEN, cls.BOS_TOKEN, cls.EOS_TOKEN, cls.UNK_TOKEN]

        tokens: List[str] = []
        tokens.append(cls.SP_TOKEN)

        # Colors
        tokens.extend(["C_W", "C_B"])

        # Pieces
        tokens.extend(["PI_P", "PI_N", "PI_B", "PI_R", "PI_Q", "PI_K"])

        # Squares
        tokens.extend([f"SQ_{sq}" for sq in cls._all_squares()])

        # Promotions
        tokens.extend(["PR_NONE", "PR_Q", "PR_R", "PR_B", "PR_N"])

        # Suffixes
        tokens.extend([
            "SUF_NONE",
            "SUF_X",
            "SUF_CHECK",
            "SUF_MATE",
            "SUF_X_CHECK",
            "SUF_X_MATE",
            "SUF_O",
            "SUF_OO",
        ])

        vocab = {tok: i for i, tok in enumerate(special + tokens)}
        return vocab

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._vocab)

    # ---------- Tokenization ----------
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a full game string. Input format is space-separated moves,
        e.g. "[BOS] WPe2e4 BPe7e5 ..."

        We emit <SP> after every "word" except EOS, so the model always sees
        a separator after moves and is in a "start-of-move" state after <SP>.
        """
        # We do NOT strip because we want predictable behavior,
        # but split() anyway collapses whitespace. That's OK.
        words = text.split()

        out: List[str] = []
        for w in words:
            if w in (self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN):
                out.append(w)
                # Put a separator after BOS as well, so pattern is: BOS <SP> MOVE...
                if w != self.EOS_TOKEN:
                    out.append(self.SP_TOKEN)
                continue

            out.extend(self._tokenize_one_move(w))
            # Always add separator after a move so evaluate.py can stop on it
            out.append(self.SP_TOKEN)

        return out

    def _tokenize_one_move(self, move: str) -> List[str]:
        """
        Parse one extended-UCI move like:
        - WPe2e4
        - BNg8f6(x)
        - WPe7e8=Q(+)
        - WKe1g1(o)
        """
        m = _MOVE_RE.match(move)
        if not m:
            return [self.UNK_TOKEN]

        color = m.group("color")   # W/B
        piece = m.group("piece")   # P/N/B/R/Q/K
        from_sq = m.group("from_sq")
        to_sq = m.group("to_sq")
        promo = m.group("promo")   # like "=Q" or None
        suffix = m.group("suffix") # like "(x+*)" or None

        toks: List[str] = []
        toks.append("C_W" if color == "W" else "C_B")
        toks.append(f"PI_{piece}")
        toks.append(f"SQ_{from_sq}")
        toks.append(f"SQ_{to_sq}")

        # Promotion token ALWAYS present (PR_NONE if absent)
        if promo is None:
            toks.append("PR_NONE")
        else:
            # promo is like "=Q"
            toks.append(f"PR_{promo[1]}")

        # Suffix token ALWAYS present
        toks.append(self._suffix_to_token(suffix))

        return toks

    def _suffix_to_token(self, suffix: Optional[str]) -> str:
        if not suffix:
            return "SUF_NONE"

        # suffix includes parentheses
        inner = suffix[1:-1]  # "x", "+", "+*", "x+", "x+*", "o", "O", ...
        if inner == "x":
            return "SUF_X"
        if inner == "+":
            return "SUF_CHECK"
        if inner == "+*":
            return "SUF_MATE"
        if inner == "x+":
            return "SUF_X_CHECK"
        if inner == "x+*":
            return "SUF_X_MATE"
        if inner == "o":
            return "SUF_O"
        if inner == "O":
            return "SUF_OO"

        # Fallback: if unknown combination appears, drop it
        return "SUF_NONE"

    # ---------- Conversions ----------
    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._vocab.get(self.UNK_TOKEN, 0))

    def _convert_id_to_token(self, index: int) -> str:
        return self._ids_to_tokens.get(index, self.UNK_TOKEN)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert tokens back to an extended-UCI string stream.

        Key constraint: evaluate.py expects generated move strings to start with
        W/B + piece letter + from + to at fixed char offsets (it slices [2:6]).
        So we must decode a move as: "WPe2e4" + optional "=Q" + optional "(x)" etc.
        And the separator token must decode to whitespace " ".
        """
        out: List[str] = []

        for tok in tokens:
            # Separator
            if tok == self.SP_TOKEN:
                out.append(" ")
                continue

            # Special tokens: keep as literal strings unless user removes them with skip_special_tokens
            if tok in (self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN):
                out.append(tok)
                continue

            # Colors
            if tok == "C_W":
                out.append("W")
                continue
            if tok == "C_B":
                out.append("B")
                continue

            # Pieces
            if tok.startswith("PI_"):
                out.append(tok.split("_", 1)[1])
                continue

            # Squares
            if tok.startswith("SQ_"):
                out.append(tok.split("_", 1)[1])
                continue

            # Promotions
            if tok == "PR_NONE":
                continue
            if tok.startswith("PR_"):
                out.append("=" + tok.split("_", 1)[1])
                continue

            # Suffixes
            if tok == "SUF_NONE":
                continue
            if tok == "SUF_X":
                out.append("(x)")
                continue
            if tok == "SUF_CHECK":
                out.append("(+)")
                continue
            if tok == "SUF_MATE":
                out.append("(+*)")
                continue
            if tok == "SUF_X_CHECK":
                out.append("(x+)")
                continue
            if tok == "SUF_X_MATE":
                out.append("(x+*)")
                continue
            if tok == "SUF_O":
                out.append("(o)")
                continue
            if tok == "SUF_OO":
                out.append("(O)")
                continue

            # Unknown token fallback
            out.append(tok)

        return "".join(out)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json",
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)

        return (vocab_file,)

    # ---------- Backward-compatible builders ----------
    @classmethod
    def build_vocab_from_dataset(
        cls,
        *args,
        **kwargs,
    ) -> "ChessTokenizer":
        """
        Kept for compatibility with train.py templates.
        Sub-token vocab is fixed, so dataset args are ignored.
        """
        return cls()
