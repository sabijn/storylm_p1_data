import numpy as np

class SyntacticDepth():
    def __init__(self, nlp):
        self.nlp = nlp

    def _walk_tree(self, node, depth):
        """
        Traverse tree
        """
        if node.n_lefts + node.n_rights > 0:
            return max(self._walk_tree(child, depth + 1) for child in node.children)
        else:
            return depth

    # function to extract maximum tree depth
    def _extract_treedepth_sent(self, text):
        """
        Convert text with spacy. 
        """
        doc = self.nlp(text)

        return max(self._walk_tree(sent.root, 0) for sent in doc.sents)

    # sum and average found treedepths per sentence
    def _average_treedepth(self, raw_story):
        """
        Sum and average treedepths per sentence
        """
        treedepths = []
        
        for l in raw_story.splitlines():
            treedepths.append(self._extract_treedepth_sent(l)) 
        
        return np.sum(treedepths) / len(treedepths)
    
    def evaluate(self, text):
        return self._average_treedepth(text)