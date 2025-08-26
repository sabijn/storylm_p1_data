from collections import Counter

# ---------- Pair extractors (plug-ins) ----------

def extract_bigrams_from_doc(doc) -> Counter:
    """
    Returns Counter of (lemma_{i-1}, lemma_i) from a spaCy Doc.
    """
    bag = Counter()
    # optional: skip punctuation/space tokens
    toks = [t for t in doc if not (t.is_space or t.is_punct)]
    for i in range(len(toks) - 1):
        bag[(toks[i].lemma_, toks[i+1].lemma_)] += 1
    return bag

def _collect_dep_pairs(node, bag: Counter):
    """
    Collect linguistically constrained pairs (dependent -> head), by lemma.
    Adjust rules as you like, but keep orientation consistent with your lexicon.
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
        _collect_dep_pairs(child, bag)

def extract_dep_pairs_from_doc(doc) -> Counter:
    """
    Returns Counter of dependency-selected (lemma1, lemma2) pairs from a Doc.
    """
    bag = Counter()
    for sent in doc.sents:
        _collect_dep_pairs(sent.root, bag)
    return bag

def extract_unigrams_from_doc(doc, *, include_punct=False, include_space=False) -> Counter:
    """
    Returns Counter of (lemma, pos).
    Excludes punctuation/space by default; flip flags to include them if desired.
    """
    bag = Counter()
    for t in doc:
        if (not include_space and t.is_space) or (not include_punct and t.is_punct):
            continue
        bag[(t.lemma_, t.pos_)] += 1   # (lemma, POS)
    return bag





