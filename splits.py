import numpy as np
from collections import defaultdict

def build_file_level_presence(y_lines, files_ids):
    y = np.asarray(y_lines)
    g = np.asarray(files_ids)
    D = y.shape[1]

    file_to_idx = defaultdict(list)
    for i, fid in enumerate(g):
        file_to_idx[fid].append(i)

    files = list(file_to_idx.keys())
    Yf = np.zeros((len(files), D), dtype=np.int64)
    sizes = np.zeros(len(files), dtype=np.int64)

    for i, fid in enumerate(files):
        idx = file_to_idx[fid]
        sizes[i] = len(idx)
        Yf[i] = (y[idx].sum(axis=0) > 0).astype(np.int64)

    return files, file_to_idx, Yf, sizes

def stratified_group_shuffle_split(
    y_lines, files_ids,
    test_frac=0.5,
    n_repeats=20,
    seed=0,
    min_presence_per_material_in_val=1,
):
    files, file_to_idx, Yf, sizes = build_file_level_presence(y_lines, files_ids)
    rng = np.random.default_rng(seed)

    n_files = len(files)
    n_val = int(round(test_frac * n_files))
    n_val = max(1, min(n_files - 1, n_val))

    total = Yf.sum(axis=0)
    target_val = total * (n_val / n_files)

    def score(val_sum):
        return np.mean(((val_sum - target_val) / (target_val + 1.0)) ** 2)

    splits = []
    for r in range(n_repeats):
        # random order, but push rare-material files earlier
        eps = 1e-9
        rarity = (Yf / (total + eps)).sum(axis=1)
        order = np.argsort(-(rarity + 0.01 * rng.random(n_files)))

        chosen = np.zeros(n_files, dtype=bool)
        val_sum = np.zeros(Yf.shape[1], dtype=float)

        # greedy fill val set
        for i in order:
            if chosen.sum() >= n_val:
                break
            new_sum = val_sum + Yf[i]
            # accept if improves score, or if we need to satisfy coverage
            if score(new_sum) <= score(val_sum):
                chosen[i] = True
                val_sum = new_sum
            else:
                # sometimes accept early to avoid getting stuck
                if chosen.sum() < max(1, n_val // 6):
                    chosen[i] = True
                    val_sum = new_sum

        # fill remaining slots with best improvement
        remaining = np.where(~chosen)[0].tolist()
        while chosen.sum() < n_val:
            best_i, best_s = None, None
            for i in remaining:
                s = score(val_sum + Yf[i])
                if best_s is None or s < best_s:
                    best_s, best_i = s, i
            chosen[best_i] = True
            val_sum += Yf[best_i]
            remaining.remove(best_i)

        # enforce minimum presence in val if feasible
        # (if total presence for a material < min_presence_per_material_in_val, cannot enforce)
        feasible = (total >= min_presence_per_material_in_val)
        if feasible.any():
            val_pres = val_sum
            if np.any((val_pres < min_presence_per_material_in_val) & feasible):
                # if violated, just skip this repeat (or you could implement swaps)
                continue

        val_files = [files[i] for i in range(n_files) if chosen[i]]
        train_files = [files[i] for i in range(n_files) if not chosen[i]]

        val_idx = np.concatenate([np.asarray(file_to_idx[f], dtype=np.int64) for f in val_files])
        train_idx = np.concatenate([np.asarray(file_to_idx[f], dtype=np.int64) for f in train_files])

        splits.append((train_idx, val_idx))

    if not splits:
        raise RuntimeError("Could not generate any split meeting the constraints. Relax constraints or reduce test_frac.")
    return splits
def file_level_presence(y, files_ids, idx,D):
    file_to_or = defaultdict(lambda: np.zeros(D, dtype=np.int64))
    for i in idx:
        fid = str(files_ids[i]).strip()
        file_to_or[fid] |= (y[i] > 0).astype(np.int64)
    Yf = np.stack(list(file_to_or.values())) if file_to_or else np.zeros((0, D), dtype=np.int64)
    return len(file_to_or), Yf.sum(axis=0)
