
class VocabularyPerplexity():
    def __init__(self, nlp):
        self.nlp = nlp
    


    def evaluate(self, text: str) -> float:
        return self._words_before_root(text)
