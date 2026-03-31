import re

class WBRAverage():
    def __init__(self, nlp):
        self.nlp = nlp
    
    @staticmethod
    def _safe_divide(numerator, denominator) -> float:
        try:
            index = numerator/denominator
        except:
            index = 0

        return index

    def _words_before_root(self, text):
        words = 0
        doc = self.nlp(text)
        sents = list(doc.sents)
        
        for sent in sents:
            for token in sent:
                if token.dep_ != 'ROOT':
                    words += 1
                else:
                    break

        return self._safe_divide(words, len(sents))

    def evaluate(self, text: str) -> float:
        text = re.sub(r'\n', ' ', text).strip() # spacy dutch is hugely impacted by newlines
        text = re.sub(' +', ' ', text)
    
        return self._words_before_root(text)

