"""Deterministic fixed-compute samplers for Step-3 causal controls."""
import hashlib
import numpy as np
from torch.utils.data import Sampler

LABELS = ["diabetic_retinopathy", "macular_edema"]
ARMS = ("C1_equal_domain", "C2_target_label", "C3_equal_domain_target_label")


def stream_seed(seed, name, index=0):
    payload = f"{seed}:{name}:{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def largest_remainder(total, weights):
    """Allocate an integer total proportionally with stable tie breaking."""
    weights = np.asarray(weights, dtype=np.float64)
    if total < 0 or weights.ndim != 1 or len(weights) == 0 or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Invalid quota request")
    raw = total * weights / weights.sum()
    result = np.floor(raw).astype(np.int64)
    remainder = int(total - result.sum())
    order = np.lexsort((np.arange(len(weights)), -(raw - result)))
    result[order[:remainder]] += 1
    return result


def even_schedule(counts, seed):
    """Spread exact category quotas through a sequence using cumulative deficits."""
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1 or len(counts) == 0 or (counts < 0).any() or counts.sum() < 1:
        raise ValueError("Invalid category counts")
    total = int(counts.sum())
    assigned = np.zeros_like(counts)
    priority = np.random.default_rng(seed).permutation(len(counts))
    rank = np.empty(len(counts), dtype=np.int64)
    rank[priority] = np.arange(len(counts))
    sequence = np.empty(total, dtype=np.int16)
    for position in range(total):
        deficit = (position + 1) * counts / total - assigned
        available = assigned < counts
        best = np.max(deficit[available])
        candidates = np.flatnonzero(available & np.isclose(deficit, best, rtol=0, atol=1e-12))
        chosen = int(candidates[np.argmin(rank[candidates])])
        sequence[position] = chosen
        assigned[chosen] += 1
    if not np.array_equal(np.bincount(sequence, minlength=len(counts)), counts):
        raise AssertionError("Schedule quota mismatch")
    return sequence


def label_key(frame):
    return frame[LABELS[0]].astype(int).astype(str) + frame[LABELS[1]].astype(int).astype(str)


def quota_spec(rows, arm, total_draws):
    if arm not in ARMS:
        raise ValueError(f"Unknown Step-3 arm: {arm}")
    domains = ("BRSET", "mBRSET")
    domain_sizes = np.array([(rows.domain == d).sum() for d in domains], dtype=np.int64)
    if (domain_sizes == 0).any():
        raise ValueError("Both domains are required")
    if arm in ("C1_equal_domain", "C3_equal_domain_target_label"):
        domain_quota = largest_remainder(total_draws, [1, 1])
    else:
        domain_quota = largest_remainder(total_draws, domain_sizes)

    target = rows[rows.domain == "mBRSET"]
    keys = ("00", "01", "10", "11")
    target_counts = np.array([(label_key(target) == key).sum() for key in keys], dtype=np.int64)
    if (target_counts == 0).any():
        raise ValueError("Target label matching requires every joint-label stratum")

    groups = []
    for domain, quota in zip(domains, domain_quota):
        subset = rows[rows.domain == domain]
        if arm == "C1_equal_domain":
            groups.append({"domain": domain, "stratum": "natural", "quota": int(quota),
                           "row_indices": subset.index.to_numpy(dtype=np.int64)})
        else:
            strata_quota = largest_remainder(int(quota), target_counts)
            subset_keys = label_key(subset)
            for key, amount in zip(keys, strata_quota):
                indices = subset.index[subset_keys == key].to_numpy(dtype=np.int64)
                if len(indices) == 0:
                    raise ValueError(f"Empty sampling cell {domain}/{key}")
                groups.append({"domain": domain, "stratum": key, "quota": int(amount),
                               "row_indices": indices})
    if sum(group["quota"] for group in groups) != total_draws:
        raise AssertionError("Total quota mismatch")
    return groups


def build_draw_indices(rows, arm, total_draws, seed):
    groups = quota_spec(rows, arm, total_draws)
    if arm == "C1_equal_domain":
        group_order = even_schedule([g["quota"] for g in groups], stream_seed(seed, arm + "_groups"))
    else:
        # Schedule domains first so C3 has exactly eight images from each domain
        # in every 16-image microbatch; then distribute label strata within the
        # occurrence positions of each domain.
        domains = ("BRSET", "mBRSET")
        domain_quota = [sum(g["quota"] for g in groups if g["domain"] == d) for d in domains]
        domain_order = even_schedule(domain_quota, stream_seed(seed, arm + "_domains"))
        group_order = np.empty(total_draws, dtype=np.int16)
        for domain_id, domain in enumerate(domains):
            positions = np.flatnonzero(domain_order == domain_id)
            group_ids = [i for i, g in enumerate(groups) if g["domain"] == domain]
            local = even_schedule([groups[i]["quota"] for i in group_ids],
                                  stream_seed(seed, arm + "_strata_" + domain))
            group_order[positions] = np.asarray(group_ids, dtype=np.int16)[local]

    draw_indices = np.empty(total_draws, dtype=np.int64)
    for group_id, group in enumerate(groups):
        positions = np.flatnonzero(group_order == group_id)
        pool = group["row_indices"]
        chosen = []
        for cycle, start in enumerate(range(0, len(positions), len(pool))):
            order = np.random.default_rng(stream_seed(seed, arm + f"_group_{group_id}", cycle)).permutation(pool)
            chosen.append(order[: min(len(pool), len(positions) - start)])
        draw_indices[positions] = np.concatenate(chosen)
    return draw_indices, groups


class ControlledBatches(Sampler):
    """Return deterministic (row index, global draw) batches with exact quotas."""
    def __init__(self, rows, arm, batch_size, start, stop, seed):
        if start < 0 or stop <= start or start % batch_size or stop % batch_size:
            raise ValueError("Invalid draw interval")
        self.batch_size, self.start, self.stop = batch_size, start, stop
        self.indices, self.groups = build_draw_indices(rows, arm, stop, seed)

    def __len__(self):
        return (self.stop - self.start) // self.batch_size

    def __iter__(self):
        for start in range(self.start, self.stop, self.batch_size):
            yield [(int(self.indices[draw]), draw) for draw in range(start, start + self.batch_size)]


def sampling_ledger(rows, arm, total_draws, seed):
    indices, groups = build_draw_indices(rows, arm, total_draws, seed)
    sampled = rows.iloc[indices].copy()
    sampled["stratum"] = label_key(sampled)
    records = []
    for domain in ("BRSET", "mBRSET"):
        for stratum in ("00", "01", "10", "11"):
            mask = (sampled.domain.to_numpy() == domain) & (sampled.stratum.to_numpy() == stratum)
            selected = indices[mask]
            unique = int(np.unique(selected).size)
            records.append({"domain": domain, "stratum": stratum, "draws": int(mask.sum()),
                            "unique_images": unique, "repeat_draws": int(mask.sum()) - unique})
    return {"arm": arm, "total_draws": int(total_draws), "records": records,
            "domain_draws": {d: int((sampled.domain == d).sum()) for d in ("BRSET", "mBRSET")},
            "positive_draws": {label: int(sampled[label].sum()) for label in LABELS},
            "all_indices_in_range": bool((indices >= 0).all() and (indices < len(rows)).all()),
            "declared_group_quotas": [{"domain": g["domain"], "stratum": g["stratum"],
                                        "quota": g["quota"], "available_images": len(g["row_indices"])}
                                       for g in groups]}
