
class AverageComponents():
    def __init__(self, nlp):
        self.nlp = nlp
    
    @staticmethod
    def _safe_divide(numerator, denominator) -> float:
        try:
            index = numerator/denominator
        except:
            index = 0

        return index

    def _extract_complements(self, story: str) -> float:
        """
        Function to average (open) clausal complements
        Returns average (float)
        """
        comps = 0

        sent_list = story.splitlines()

        for line in sent_list:
            doc = self.nlp(line)
            
            for token in doc:
                if token.dep_ == 'ccomp':
                    # clausal complement ('I think that he is lying')
                    comps += 1
                elif token.dep_ == 'xcomp':
                    # open clausal complement ('He made her cry')
                    comps += 1
        
        return self._safe_divide(comps, len(sent_list))

    def evaluate(self, text: str) -> float:
        return self._extract_complements(text)