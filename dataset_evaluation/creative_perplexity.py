import pandas as pd
from pathlib import Path
from collections import Counter
import numpy as np

class CreativePerplexity():
    def __init__(self, nlp, mode, unigram, bigram, constrained_bigram):
        self.nlp = nlp
        self.mode = mode
        self.unigram = self._load_reference_corpora(unigram)
        self.bigram = self._load_reference_corpora(bigram)
        self.constrained_bigram = self._load_reference_corpora(constrained_bigram)

    def _load_reference_corpora(self, data_path: str) -> pd.DataFrame:
        if Path(data_path).exists():
            return pd.read_csv(data_path)
        
        raise Exception(f'Reference corpora are not registered.')
    
    def _extract_unigrams(self, story, *, with_pos=False,
                     include_punct=False, include_space=False, exclude_pos=None) -> Counter:
        """
        Return Counter of (lemma,) or (lemma,pos) from the story.
        """
        doc = self.nlp(story)
        bag = Counter()
        for t in doc:
            if not include_space and t.is_space: 
                continue
            if not include_punct and t.is_punct: 
                continue
            if exclude_pos and t.pos_ in exclude_pos:
                continue
            if with_pos:
                bag[(t.lemma_, t.pos_)] += 1
            else:
                bag[(t.lemma_,)] += 1
        return bag

    def _extract_bigrams(self, story):
        """
        Return Counter of (lemma_{i-1}, lemma_i) from the story.
        """
        doc = self.nlp(story)
        pairs = [(doc[i].lemma_, doc[i+1].lemma_) for i in range(len(doc)-1)]

        return Counter(pairs)

    def _collect_dep_pairs(self, node, bag: Counter):
        """
        Collect linguistically constrained pairs.
        Here: (child -> head) for selected relations.
        Adjust the rules to your needs, but keep orientation consistent with lexicon.
        """
        # Verbal heads: subject/object relations
        if node.pos_ in ('VERB', 'AUX'):
            for child in node.children:
                if child.dep_ in ('nsubj', 'nsubj:pass', 'obj'):
                    bag[(child.lemma_, node.lemma_)] += 1

        # Nominal heads: adjectival modifiers
        if node.pos_ == 'NOUN':
            for child in node.children:
                if child.dep_ == 'amod' and child.pos_ == 'ADJ':
                    bag[(child.lemma_, node.lemma_)] += 1

        # Recurse
        for child in node.children:
            self._collect_dep_pairs(child, bag)

    def _extract_dep_pairs(self, story):
        """
        Return Counter of linguistically constrained (lemma1, lemma2) pairs.
        Current orientation: (dependent lemma, head lemma).
        """
        bag = Counter()
        for sent in self.nlp(story).sents:
            self._collect_dep_pairs(sent.root, bag)
        return bag

    def _compute_unigram_perplexity_from_counts(
        self,
        story_counts: Counter,
        lexicon,
        *,
        cols=("lemma",),                # or ("lemma","pos")
        alpha: float = 1.0,
        return_frame: bool = False,
    ):
        """
        Unigram PPL with Laplace smoothing:
        p(w) = (c_corpus(w) + alpha) / (C + alpha*V)
        Weighted by story token counts.
        """
        if not story_counts:
            return (None, pd.DataFrame(columns=[*cols,"count","freq"])) if return_frame else None

        story_df = pd.DataFrame([(*k, c) for k, c in story_counts.items()],
                                columns=[*cols, "count"]) \
                .merge(lexicon, how="left", on=list(cols))
        story_df["freq"] = story_df["freq"].fillna(0).astype(np.int64)

        C = int(lexicon["freq"].sum())                                # total tokens in reference
        V = int(lexicon.drop(columns=["freq"]).drop_duplicates().shape[0]) or 1
        denom = C + alpha * V

        p = (story_df["freq"] + alpha) / denom
        logp = np.log(p)

        N = int(story_df["count"].sum())
        ppl = float(np.exp(-(logp * story_df["count"]).sum() / N))

        if return_frame:
            out = story_df.copy()
            out["prob"] = p
            return ppl, out[[*cols, "count", "freq", "prob"]]
        return ppl


    def _compute_perplexity_from_counts(
        self,
        story_counts: Counter,
        lexicon,
        *,
        alpha: float = 1.0,
        vocab: str = "global",   # "global" or "per_history"
        return_frame: bool = False
    ):
        """
        Compute perplexity for (lemma1, lemma2) counts using conditional bigram probabilities with Laplace smoothing.

        lexicon: DataFrame with ['lemma1', 'lemma2', 'freq'] from the reference corpus.
        alpha: Laplace smoothing strength (α=1 is add-one).
        vocab:
        - "global": use |{lemma2}| as V for all histories (textbook Laplace).
        - "per_history": use |{lemma2: seen after lemma1}| as V(lemma1); falls back to global V when unknown.
        return_frame: also return a diagnostic DataFrame with probs and components.
        """
        if not story_counts:
            return (None, pd.DataFrame(columns=['lemma1','lemma2','count'])) if return_frame else None

        # Story pairs + counts
        story_df = (
            pd.DataFrame([(l1, l2, c) for (l1, l2), c in story_counts.items()],
                        columns=['lemma1','lemma2','count'])
            .merge(lexicon, how='left', on=['lemma1','lemma2'])
        )
        story_df['freq'] = story_df['freq'].fillna(0)

        # Denominators: totals per history (c(lemma1))
        hist_totals = lexicon.groupby('lemma1')['freq'].sum().rename('hist_total')
        story_df = story_df.merge(hist_totals, how='left', on='lemma1')
        story_df['hist_total'] = story_df['hist_total'].fillna(0)

        # Vocabulary sizes
        global_V = int(lexicon['lemma2'].nunique()) or 1
        if vocab == "per_history":
            V_by_hist = lexicon.groupby('lemma1')['lemma2'].nunique().rename('hist_vocab')
            story_df = story_df.merge(V_by_hist, how='left', on='lemma1')
            story_df['hist_vocab'] = story_df['hist_vocab'].fillna(global_V)
        else:
            story_df['hist_vocab'] = global_V

        # Smoothed conditional probabilities p(lemma2 | lemma1)
        denom = story_df['hist_total'] + alpha * story_df['hist_vocab']

        denom = denom.replace(0, np.finfo(float).tiny)  # Avoid log(0)
        probs = (story_df['freq'] + alpha) / denom
        logp = np.log(probs)

        # Perplexity over total pair count (weights by story counts)
        N = int(story_df['count'].sum())
        if N == 0:
            return (None, story_df) if return_frame else None

        avg_logp = (logp * story_df['count']).sum() / N
        ppl = float(np.exp(-avg_logp))

        if return_frame:
            out = story_df.copy()
            out['prob'] = probs
            return ppl, out[['lemma1','lemma2','count','freq','hist_total','hist_vocab','prob']]
        
        return ppl

    def _perplexity_for_creativity(
        self,
        story: str,
        *,
        alpha: float,
        vocab: str,    # for bigrams only: "global" or "per_history"
        with_pos_unigram: bool,   # unigram extractor option
        unigram_cols,        # or ("lemma","pos") to match your lexicon
        return_frame: bool,
    ):
        """
        General entry point. Chooses the extractor and computes perplexity.
        IMPORTANT: Your lexicon columns must match the extractor/cols:
        - unigram_cols for 'unigram'
        - ['lemma1','lemma2','freq'] for 'bigram'/'dep'
        """
        if self.mode == "bigram":
            counts = self._extract_bigrams(story)
            return self._compute_perplexity_from_counts(
                counts, self.bigram, alpha=alpha, vocab=vocab, return_frame=return_frame
            )
        elif self.mode == "dep":
            counts = self._extract_dep_pairs(story)
            return self._compute_perplexity_from_counts(
                counts, self.constrained_bigram, alpha=alpha, vocab=vocab, return_frame=return_frame
            )
        elif self.mode == "unigram":
            counts = self._extract_unigrams(story, with_pos=with_pos_unigram)
            return self._compute_unigram_perplexity_from_counts(
                counts, self.unigram, cols=unigram_cols, alpha=alpha, return_frame=return_frame
            )
        else:
            raise ValueError("mode must be 'unigram', 'bigram', or 'dep'")
    
    def evaluate(self, text: str, vocab="per_history", alpha=1.0, with_pos_unigram=True,
                 unigram_cols=("lemma","pos"), return_frame=False
                 ) -> float:
        return self._perplexity_for_creativity(text,
                                               vocab=vocab,
                                               alpha=alpha,
                                               with_pos_unigram=with_pos_unigram,
                                               unigram_cols=unigram_cols,
                                               return_frame=return_frame
                                               )