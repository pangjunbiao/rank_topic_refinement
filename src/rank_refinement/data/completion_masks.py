"""Occurrence partitions and encoder views contain no held-out target tokens."""
import random
from ..runtime.seeds import keyed_seed


def make_completion(group_id, tokens, word_to_id, policy):
    positions = [i for i, token in enumerate(tokens) if token in word_to_id]
    ids = [word_to_id[tokens[i]] for i in positions]
    eligible = len(ids) >= int(policy["minimum_in_vocabulary_tokens"])
    if not eligible:
        return {"group_id": group_id, "eligible": False, "reason": "fewer_than_two_in_vocabulary_tokens",
                "observed_positions": [], "target_positions": [], "observed_ids": [], "target_ids": [], "observed_text": ""}
    shuffled = list(range(len(ids)))
    random.Random(keyed_seed(policy["seed"], "completion", group_id)).shuffle(shuffled)
    n_observed = max(1, min(len(ids) - 1, int(len(ids) * policy["observed_fraction"])))
    chosen = set(shuffled[:n_observed])
    observed = [i for i in range(len(ids)) if i in chosen]
    target = [i for i in range(len(ids)) if i not in chosen]
    return {"group_id": group_id, "eligible": True, "reason": None,
            "observed_positions": [positions[i] for i in observed], "target_positions": [positions[i] for i in target],
            "observed_ids": [ids[i] for i in observed], "target_ids": [ids[i] for i in target],
            "observed_text": " ".join(tokens[positions[i]].replace("_", "") for i in observed)}
