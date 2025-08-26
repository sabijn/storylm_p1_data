import spacy
import pandas as pd
import logging

logger = logging.getLogger(__name__)

from .syntactic_depth import SyntacticDepth
from .average_components import AverageComponents
from .wbr_average import WBRAverage
from .lexical_d import LexicalDiversity
from .dependency_distance import DependencyDistance
from .grammaticality import Grammaticality

class EvaluationFramework:
    def __init__(self, language, 
                 pos_unigram: pd.DataFrame = None, 
                 pos_bigram: pd.DataFrame = None, 
                 pos_trigram: pd.DataFrame = None,
                 ref_unigram: pd.DataFrame = None, 
                 ref_bigram: pd.DataFrame = None, 
                 ref_ling_constrained: pd.DataFrame = None,):
        """
        Parameters:
        @language (str): language of the reference corpora and the text to be analysed.
        @pos_unigram (pandas dataframe): reference corpus containg unigram POS frequencies.
        @pos_bigram (pandas dataframe): reference corpus containg bigram POS frequencies.
        @pos_trigram (pandas dataframe): reference corpus containg trigram POS frequencies.
        @ref_unigram (pandas dataframe): reference corpus containg unigram frequencies.
        @ref_bigram (pandas dataframe): reference corpus containg bigram frequencies.
        @ref_ling_constrained (pandas dataframe): reference corpus containg linguistically constrained frequencies.
        """
        self.pipeline = []

        if language == 'nl':
            #TODO: add specific function to download the correct thingie, otherwise ask user to download it
            self.nlp = spacy.load("nl_core_news_lg") 
        elif language == 'en':
            self.nlp = spacy.load("en_core_web_sm") 
        else:
            logger.info('No language specified, defaulting to English')
            self.nlp = spacy.load("en_core_web_sm")
        
        self.pos_unigram = pos_unigram
        self.pos_bigram = pos_bigram
        self.pos_trigram = pos_trigram

        self.ref_unigram = ref_unigram
        self.ref_bigram = ref_bigram
        self.ref_ling_constrained = ref_ling_constrained

    def add_pipe(self, name, component=None):
        if name == 'grammaticality' and (self.pos_unigram == None or self.pos_bigram == None or self.pos_trigram == None):
            raise f'The grammaticality component needs the 1/2/3-gram corpora. Initialize the EvaluationFramework with these corpora.'
        
        if name == 'creative_perplexity' and (self.ref_unigram == None or self.ref_bigram == None or self.ref_ling_constrained == None):
            raise f'The creative perplexity component needs the 1/2/3-gram corpora. Initialize the EvaluationFramework with these corpora.'

        if component is None:
            component = self._get_builtin_component(name)

        self.pipeline.append((name, component))

    def change_order_pipeline(self):
        raise NotImplementedError

    def _get_builtin_component(self, name):
        # Register your built-in components here
        self.registry = {
            "syntactic_depth": SyntacticDepth(self.nlp),
            "average_components": AverageComponents(self.nlp),
            "wbr_average": WBRAverage(self.nlp),
            "lexical_diversity": LexicalDiversity(),
            "dependency_distance": DependencyDistance(self.nlp),
            "grammaticality":  Grammaticality(self.nlp, self.pos_unigram, self.pos_bigram, self.pos_trigram)

        }

        if name not in self.registry:
            raise ValueError(f"Component '{name}' not found.")
        
        return self.registry[name]
    
    def register_component(self, name, component):
        if name not in self.registry:
            self.registry[name] = component
        else:
            return 1

    def run_component(self, name, data, *args, **kwargs):
        for comp_name, component in self.pipeline:
            if comp_name == name:
                return component.evaluate(data, *args, **kwargs)
        raise ValueError(f"Component '{name}' not found in the pipeline.")
    
    def run_pipeline(self, data, *args, **kwargs):
        results = {}
        for name, component in self.pipeline:
            results[name] = component.evaluate(data, *args, **kwargs)
        return results
    
    def run_pipeline_on_df(self, df, column_name, *args, **kwargs):

        for name, component in self.pipeline:
            print(f'Running {name}')
            df[name] = df.apply(lambda row: component.evaluate(row[column_name], *args, **kwargs), axis = 1) 

        return df