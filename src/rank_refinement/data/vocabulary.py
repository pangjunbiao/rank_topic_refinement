"""Fitting-only phrase and vocabulary models, reusable inside each fold."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from ..runtime.artifacts import digest, write_csv, write_json


def fit_features(documents, fit_group_ids, phrase_policy, vocabulary_policy, destination):
    from gensim.models.phrases import Phrases, Phraser
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    fit_ids = set(fit_group_ids)
    fitting = [documents[g]["base_tokens"] for g in sorted(fit_ids)]
    if not fitting:
        raise ValueError("No fitting groups for feature preparation.")
    phraser = None
    if phrase_policy["enabled"]:
        model = Phrases(fitting, min_count=int(phrase_policy["min_count"]), threshold=float(phrase_policy["threshold"]),
                        scoring=phrase_policy["scoring"], delimiter=phrase_policy["delimiter"])
        phraser = Phraser(model)
        phraser.save(str(destination / "phrases.gensim"), separately=[])
        phrase_rows = [{"phrase": key, "score": float(value)} for key, value in sorted(phraser.phrasegrams.items())]
    else:
        phrase_rows = []
    transformed = {g: list(phraser[documents[g]["base_tokens"]]) if phraser is not None else list(documents[g]["base_tokens"])
                   for g in sorted(documents)}
    frequency = Counter()
    token_frequency = Counter()
    for g in sorted(fit_ids):
        frequency.update(set(transformed[g]))
        token_frequency.update(transformed[g])
    n = len(fit_ids)
    minimum, maximum = vocabulary_policy["min_document_frequency"], vocabulary_policy["max_document_fraction"] * n
    words = sorted(w for w, count in frequency.items() if count >= minimum and count <= maximum)
    if not words:
        raise ValueError("Fitting-only vocabulary is empty.")
    word_to_id = {word: i for i, word in enumerate(words)}
    metadata = {"tokens": words, "word_to_id": word_to_id, "fit_groups": n,
                "fit_group_ids_sha256": digest(sorted(fit_ids)), "document_frequency": [frequency[w] for w in words],
                "token_frequency": [token_frequency[w] for w in words], "vocabulary_policy": vocabulary_policy,
                "phrase_policy": phrase_policy, "phrase_count": len(phrase_rows), "vocabulary_sha256": digest(words)}
    write_json(destination / "vocabulary.json", metadata)
    write_csv(destination / "vocabulary.csv", [{"word_id": i, "word": w, "fitting_group_df": frequency[w], "fitting_token_count": token_frequency[w]} for i, w in enumerate(words)],
              ["word_id", "word", "fitting_group_df", "fitting_token_count"])
    write_csv(destination / "learned_phrases.csv", phrase_rows, ["phrase", "score"])
    filter_rows = [{"word": w, "fitting_group_df": frequency[w], "fitting_token_count": token_frequency[w],
                    "decision": "retained" if w in word_to_id else "below_min_df" if frequency[w] < minimum else "above_max_fraction"}
                   for w in sorted(frequency, key=lambda w: (-frequency[w], w))]
    write_csv(destination / "vocabulary_filter_audit.csv", filter_rows, ["word", "fitting_group_df", "fitting_token_count", "decision"])
    return transformed, metadata
