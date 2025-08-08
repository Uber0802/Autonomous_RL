# utils/tokenizer_cfg.py
from typing import List

def configure_tokenizer(tokenizer) -> None:
    """
    Make tokenizer behaviour deterministic & add <act_i> tokens.
    This must be called **once** on every tokenizer object
    (train / eval / data‑prep) before you use it.
    """
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side   = "right"
    tokenizer.truncation_side = "left"

    act_tokens: List[str] = [f"<act_{i}>" for i in range(7)]
    _ = tokenizer.add_tokens(
        [tok for tok in act_tokens if tok not in tokenizer.get_vocab()],
        special_tokens=True,
    )
    