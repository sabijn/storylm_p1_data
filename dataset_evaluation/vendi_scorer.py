

# vendi_score_component.py
from typing import List, Sequence
from vendi_score import text_utils
import re

class VendiScore:
    """
    Compute Vendi scores per text element.

    Modes
    -----
    - method="ngram": vendi_score.text_utils.ngram_vendi_score
    - method="embeddings": vendi_score.text_utils.embedding_vendi_score

    Parameters
    ----------
    nlp : spaCy Language
        Used for sentence segmentation.
    method : {"ngram","embeddings"}
        Which variant to run in evaluate().
    ns : Sequence[int], optional (ngram mode)
        n-gram sizes to consider, e.g., [1, 2]. Default: (1, 2).
    model_path : str, optional (embeddings mode)
        Model identifier to use in embedding_vendi_score. Default: "bert-base-uncased".
    min_items : int
        Minimum number of sentences required to compute a score. If fewer, returns 0.0.
    """

    def __init__(
        self,
        nlp,
        method: str = "ngram",
        ns: Sequence[int] = (1, 2),
        model_path: str = "bert-base-uncased",
        min_items: int = 2,
    ):
        method = method.lower().strip()
        if method not in {"ngram", "embeddings"}:
            raise ValueError("method must be 'ngram' or 'embeddings'")
        self.nlp = nlp
        self.method = method
        self.ns = tuple(ns)
        self.model_path = model_path
        self.min_items = int(min_items)
    

    def evaluate(self, text: str) -> float:
        text = re.sub(r'\n', ' ', text).strip() # spacy dutch is hugely impacted by newlines
        text = re.sub(' +', ' ', text)
    
        sents = list(map(str, list(self.nlp(text).sents)))
        if len(sents) < self.min_items:
            return 0.0

        try:
            if self.method == "ngram":
                return float(text_utils.ngram_vendi_score(sents, ns=list(self.ns)))
            else:  # embeddings
                return float(
                    text_utils.embedding_vendi_score(sents, model_path=self.model_path)
                )
        except Exception:
            return 0.0
