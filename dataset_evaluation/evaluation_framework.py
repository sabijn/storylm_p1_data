import spacy
import logging

logger = logging.getLogger(__name__)

from .syntactic_depth import SyntacticDepth
from .average_components import AverageComponents
from .wbr_average import WBRAverage
from .lexical_d import LexicalDiversity
from .dependency_distance import DependencyDistance

class EvaluationFramework:
    def __init__(self, language):
        self.pipeline = []

        if language == 'nl':
            # add specific function to download the correct thingie, otherwise ask user to download it
            self.nlp = spacy.load("nl_core_news_lg") 
        elif language == 'en':
            self.nlp = spacy.load("en_core_news_lg") 
        else:
            logger.info('No language specified, defaulting to English')
            self.nlp = spacy.load("en_core_news_lg") 

    def add_pipe(self, name, component=None):
        if component is None:
            component = self._get_builtin_component(name)

        self.pipeline.append((name, component))

    def change_order_pipeline(self):
        raise NotImplementedError

    def _get_builtin_component(self, name):
        # Register your built-in components here
        registry = {
            "syntactic_depth": SyntacticDepth(self.nlp),
            "average_components": AverageComponents(self.nlp),
            "wbr_average": WBRAverage(self.nlp),
            "lexical_diversity": LexicalDiversity(),
            "dependency_distance": DependencyDistance(self.nlp)
        }

        if name not in registry:
            raise ValueError(f"Component '{name}' not found.")
        
        return registry[name]
    
    def register_component(self):
        raise NotImplementedError

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
    

