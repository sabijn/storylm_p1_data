

class CreativePerplexity():
    def __init__():
        pass

    def _unigram_freqs(self):
        raise NotImplementedError
    
    def _bigram_freqs(self):
        raise NotImplementedError

    def _linguistic_constrained_freqs(self):
        raise NotImplementedError
    
    def _calculate_pp(self):
        raise NotImplementedError
    
    def evaluate(self, text: str) -> float:
        return self._calculate_pp(text)