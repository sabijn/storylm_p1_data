import pandas as pd
import numpy as np

from pathlib import Path
from collections import Counter
from tqdm.notebook import tqdm

from typing import Iterable, Callable, Optional, Sequence

def build_lexicon(
    texts: Iterable[str],
    nlp,
    extractor: Callable,                 # one of the extract_*_from_doc functions
    *,
    batch_size: int = 100,
    n_process: int = 1,
    flush_every: int = 100_000,          # flush intermediate counts to disk every N pairs (optional)
    cache_path: Optional[Path] = None,   # if provided, write partials; final CSV always written by save_lexicon()
    key_names: Sequence[str] = ("lemma1","lemma2")
) -> Counter:
    """
    Streams texts through spaCy, applies 'extractor' to each Doc, and aggregates a Counter of (lemma1, lemma2) -> freq.
    """
    agg = Counter()
    seen_since_flush = 0

    # Faster: use nlp.pipe to batch & parallelize
    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size, n_process=n_process), desc="Building lexicon"):
        bag = extractor(doc)
        agg.update(bag)
        seen_since_flush += sum(bag.values())

        if cache_path and seen_since_flush >= flush_every:
            # dump a partial (append-mode parquet/csv could be used; here we overwrite a temp snapshot)
            _save_counter_as_csv(agg, cache_path)
            seen_since_flush = 0

    return agg

def _save_counter_as_csv(
    counter: Counter,
    path: Path,
    key_names: Sequence[str],
) -> pd.DataFrame:
    """
    Save a Counter with tuple keys to CSV with columns [*key_names, 'freq'].
    Works for unigrams+POS too by passing key_names=('lemma','pos').
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    expected_len = len(key_names)
    for k, v in counter.items():
        if not isinstance(k, tuple):
            k = (k,)
        if len(k) != expected_len:
            raise ValueError(
                f"Key length {len(k)} != len(key_names) {expected_len}. Key={k}, key_names={key_names}"
            )
        rows.append((*k, v))

    df = pd.DataFrame(rows, columns=[*key_names, "freq"])
    if not df.empty:
        df["freq"] = df["freq"].astype(np.int64)

    df.to_csv(path, index=False)
    return df


def save_lexicon(counter: Counter, out_path: Path, key_names: Sequence[str]) -> pd.DataFrame:
    _save_counter_as_csv(counter, out_path, key_names)
    return pd.read_csv(out_path)


def load_or_build_lexicon(
    out_path: Path,
    builder_fn: Callable[[], Counter],
    key_names: Sequence[str] = ("lemma1","lemma2"),
) -> pd.DataFrame:
    """
    If 'out_path' exists, load CSV; otherwise call builder_fn(), save, and return the DataFrame.
    """
    if out_path.exists():
        df = pd.read_csv(out_path)

        expected = {*key_names, "freq"}
        if set(df.columns) != expected:
            raise Exception(f'The old csv has not the correct columns. Check if you are reading in the right csv.')
        return df
    else:
        counter = builder_fn()
        return save_lexicon(counter, out_path, key_names=key_names)