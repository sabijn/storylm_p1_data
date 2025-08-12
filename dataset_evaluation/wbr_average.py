
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
        for s in text.splitlines():
            doc = self.nlp(s)

            for token in doc:
                if token.dep_ != 'ROOT':
                    words += 1
                else:
                    break

        return self._safe_divide(words, len(text.splitlines()))

    def evaluate(self, text: str) -> float:
        return self._words_before_root(text)

