import re
from nltk import word_tokenize

class DependencyDistance():
    def __init__(self, nlp):
        self.nlp = nlp

    def _dep_dist(self, text):
        """
        This is a custom implementation of DD based on the textdescriptives implementation of TD
        Yet, we want to split on our own utterance markers, not on spaCy's built-in senter
        We found out that it doesn't work optimal
        So here we calculate dependency distance according to the implementation of Liu (2008)
        """
        doc = self.nlp(text)

        sent_list = re.split(r'\.|\?', text)
        
        sent_list = [sent for sent in sent_list if sent not in ['Einde', ' Einde', 'einde', ' einde']] #-- exclude 1-word sentences with DD of 0
        # print(sent_list) #-- if you want to see result of splitting

        DD = [] # container for all abs. dependency distances
        s = len(sent_list) # no of sents
        # print(f'Number of sents: {s}')
        w = [] # no. of words

        s_len = [] #-- curious what average utt length is

        for sent in sent_list:
            #print(sent) #-- see the result of splitting on . and ?
            w.append(len(word_tokenize(sent, language='dutch')))

            #print(w) #-- see container with all sent lengths  
        
            doc = self.nlp(sent)
            for token in doc:
                if token.dep_ != "ROOT": #-- included some print statements as sanity checks
                    #print(token.head, token) #-- inspect pair
                    #print(f'Token in question: {token}') #-- inspect focus token
                    #print(f'Position of the head: {token.head.i}') #-- inspect head position
                    #print(f'Position of the token: {token.i}') #-- inspect focus token position
                    
                    dep_dist = abs(token.head.i - token.i)
                    #print(f'Abs. dep dist: {dep_dist}') #-- add abs dist to list

                    DD.append(dep_dist)
        
        DD = [distance for distance in DD if distance != 0] #-- drop DDs for whitespace, which are zero, and not included in w
        
        # print(DD) #-- see list with all abs dists

        return (1/(sum(w)-s)) * sum(DD)

    def evaluate(self, text: str) -> float:
        return self._dep_dist(text)