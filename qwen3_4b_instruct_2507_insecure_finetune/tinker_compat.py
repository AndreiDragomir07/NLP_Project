"""Compatibility patches for third-party tokenizer code loaded by Tinker."""

from __future__ import annotations


def install_transformers_compat_patches() -> None:
    """Install narrowly scoped compatibility shims before loading remote tokenizers.

    Some remote tokenizer implementations import ``bytes_to_unicode`` from
    ``transformers.models.gpt2.tokenization_gpt2``. Transformers 5.x no longer
    exports that helper from the GPT-2 tokenizer module, so we provide the
    standard GPT-2 implementation at the expected import location.
    """

    try:
        from transformers.models.gpt2 import tokenization_gpt2
    except Exception:
        return

    if hasattr(tokenization_gpt2, "bytes_to_unicode"):
        return

    def bytes_to_unicode() -> dict[int, str]:
        bs = list(range(ord("!"), ord("~") + 1))
        bs += list(range(ord("¡"), ord("¬") + 1))
        bs += list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return dict(zip(bs, (chr(n) for n in cs), strict=True))

    tokenization_gpt2.bytes_to_unicode = bytes_to_unicode
