

# vendi_score_component.py
from typing import List, Sequence
from diversity import homogenization_score

class SelfBleuScore:


    def __init__(
        self,
        nlp,
        model_path: str = "bert-base-uncased",
        min_items: int = 2,
    ):
        self.nlp = nlp
        self.model_path = model_path
        self.min_items = int(min_items)
    

    def evaluate(self, text: str) -> float:
        sents = list(map(str, list(self.nlp(text).sents)))
        if len(sents) < self.min_items:
            return 0.0

        try:
            return homogenization_score(sents, measure='bleu', verbose=False, model=self.model_path)
        except Exception:
            return 0.0
