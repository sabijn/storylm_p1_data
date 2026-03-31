import pandas as pd
from pathlib import Path
from collections import Counter
import numpy as np
from simplemma import text_lemmatizer

class CreativePerplexity():
    def __init__(self, nlp, language, mode, unigram, bigram, constrained_bigram):
        self.nlp = nlp
        self.mode = mode
        self.language = language
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
                if child.dep_ in ('nsubj', 'nsubj:pass', 'obj') and child.text.isalnum():
                    child_lemma = text_lemmatizer(child.text, lang=self.language)[0]
                    node_lemma = text_lemmatizer(node.text, lang=self.language)[0]
                    bag[(child_lemma, node_lemma)] += 1
        # Nominal heads: adjectival modifiers
        if node.pos_ == 'NOUN' and node.text.isalnum():
            for child in node.children:
                if child.dep_ == 'amod' and child.pos_ == 'ADJ' and child.text.isalnum():
                    child_lemma = text_lemmatizer(child.text, lang=self.language)[0]
                    node_lemma = text_lemmatizer(node.text, lang=self.language)[0]
                    bag[(child_lemma, node_lemma)] += 1

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
        lex: pd.DataFrame,       
        *,
        smoothing: str = "kneser_ney",   
        alpha: float = 0.1,              
        unk_token: str = "<unk>",
        return_frame: bool = False
    ):
        
        # Fixed vocab for lemma2 (after cutoff)
        vocab2 = set(lex['lemma2'].unique())
        global_V = len(vocab2)

        # ---------- 2) Prepare story pairs and map OOV lemma2 → <unk> ----------
        if not story_counts:
            return (None, pd.DataFrame(columns=['lemma1','lemma2','count'])) if return_frame else None

        def map_l2(l2):
            return l2 if l2 in vocab2 else unk_token

        story_pairs = [ (l1, map_l2(l2), c) for (l1, l2), c in story_counts.items() ]
        story_df = pd.DataFrame(story_pairs, columns=['lemma1','lemma2','count'])

        # Merge training counts for these pairs
        story_df = story_df.merge(lex, how='left', on=['lemma1','lemma2'])
        story_df['freq'] = story_df['freq'].fillna(0).astype(int)

        # Totals per history
        hist_totals = lex.groupby('lemma1', as_index=False)['freq'].sum().rename(columns={'freq':'hist_total'})
        story_df = story_df.merge(hist_totals, how='left', on='lemma1')
        story_df['hist_total'] = story_df['hist_total'].fillna(0).astype(int)

        # ---------- 3A) Kneser–Ney smoothing ----------
        if smoothing == "kneser_ney":
            # Continuation counts (#distinct w1 per w2)
            cont_df = (
                lex[lex['freq'] > 0]
                .groupby('lemma2', as_index=False)['lemma1']
                .nunique()
                .rename(columns={'lemma1':'cont_count'})
            )
            cont_df = pd.DataFrame({'lemma2': list(vocab2)}).merge(cont_df, how='left', on='lemma2')
            cont_df['cont_count'] = cont_df['cont_count'].fillna(0).astype(int)

            total_types = int((lex['freq'] > 0).sum())
            cont_df['p_cont'] = np.where(
                total_types > 0,
                cont_df['cont_count'] / total_types,
                1.0 / global_V
            )

            story_df = story_df.merge(cont_df[['lemma2','p_cont']], how='left', on='lemma2')
            story_df['p_cont'] = story_df['p_cont'].fillna(1.0 / global_V)

            # Distinct followers per history
            T_df = (
                lex[lex['freq'] > 0]
                .groupby('lemma1', as_index=False)['lemma2']
                .nunique()
                .rename(columns={'lemma2':'T_followers'})
            )
            story_df = story_df.merge(T_df, how='left', on='lemma1')
            story_df['T_followers'] = story_df['T_followers'].fillna(0).astype(int)

            # Discount D
            counts = lex['freq'].values
            N1 = int(np.sum(counts == 1))
            N2 = int(np.sum(counts == 2))
            D = (N1 / (N1 + 2 * N2)) if (N1 + 2 * N2) > 0 else 0.75

            hist = story_df["hist_total"].replace(0, np.nan)  # avoid division by zero
            lam = (D * story_df["T_followers"]) / hist
            lam = lam.fillna(1.0)  # unseen history → full backoff

            hist_safe = story_df['hist_total'].replace(0, np.nan)
            first_term = np.maximum(story_df['freq'] - D, 0.0) / hist_safe
            first_term = first_term.fillna(0.0)

            probs = first_term + lam * story_df['p_cont']
            probs = np.maximum(probs, np.finfo(float).tiny)

            diag_cols = ['lemma1','lemma2','count','freq','hist_total','T_followers','p_cont']
            story_df_out = story_df[diag_cols].copy()
            story_df_out['prob'] = probs

        # ---------- 3B) Add-α smoothing (global) ----------
        elif smoothing == "add_alpha":
            uni = lex.groupby('lemma2', as_index=False)['freq'].sum().rename(columns={'freq':'uni_count'})
            uni = pd.DataFrame({'lemma2': list(vocab2)}).merge(uni, how='left', on='lemma2')
            uni['uni_count'] = uni['uni_count'].fillna(0).astype(int)
            total_uni = int(uni['uni_count'].sum())
            uni['p_unigram'] = (uni['uni_count'] + alpha) / (total_uni + alpha * global_V)

            story_df = story_df.merge(uni[['lemma2','p_unigram']], how='left', on='lemma2')

            denom = story_df['hist_total'] + alpha * global_V
            denom = denom.replace(0, np.nan)  # unseen histories → NaN
            bigram_probs = (story_df['freq'] + alpha) / denom
            probs = bigram_probs.fillna(story_df['p_unigram'])  # backoff to unigram
            probs = np.maximum(probs, np.finfo(float).tiny)

            diag_cols = ['lemma1','lemma2','count','freq','hist_total']
            story_df_out = story_df[diag_cols].copy()
            story_df_out['prob'] = probs

        else:
            raise ValueError("smoothing must be 'kneser_ney' or 'add_alpha'")

        # ---------- 4) Compute perplexity ----------
        N = int(story_df['count'].sum())
        if N == 0:
            return (None, story_df_out) if return_frame else None

        avg_logp = (np.log(probs) * story_df['count']).sum() / N
        ppl = float(np.exp(-avg_logp))

        if return_frame:
            return ppl, story_df_out
        return ppl

    def _perplexity_for_creativity(
        self,
        story: str,
        *,
        alpha: float,
        vocab: str,
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
                counts, self.constrained_bigram, alpha=alpha, return_frame=return_frame
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
                                               return_frame=return_frame)