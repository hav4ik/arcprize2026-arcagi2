import json
import hashlib
import math
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from transformers import AutoTokenizer


TRAIN_PREDICTION_KEY_MARKER = "train"


def train_prediction_key(task_key, train_index):
    """Return the stable pseudo-test key for one held-out training pair."""
    return f"{task_key}_{TRAIN_PREDICTION_KEY_MARKER}{int(train_index)}"


def parse_train_prediction_key(key):
    """Return ``(task_key, train_index)`` for a training diagnostic key."""
    base_key = key.split(".", 1)[0]
    marker = f"_{TRAIN_PREDICTION_KEY_MARKER}"
    task_key, separator, index_text = base_key.rpartition(marker)
    if not separator or not task_key or not index_text.isdigit():
        return None
    return task_key, int(index_text)


def convert_grid_to_string(grid) -> str:
    # Fast path for ordinary ARC grids: one ASCII digit per cell, rows joined by
    # newlines. Produces exactly the string of the per-cell loop below.
    if isinstance(grid, (list, np.ndarray)):
        try:
            cells = np.asarray(grid)
        except ValueError:
            cells = None
        if (
            cells is not None
            and cells.ndim == 2
            and cells.size
            and cells.dtype.kind in "iu"
            and cells.min() >= 0
            and cells.max() <= 9
        ):
            rows = np.empty((cells.shape[0], cells.shape[1] + 1), dtype=np.uint8)
            rows[:, :-1] = cells + 48
            rows[:, -1] = 10
            return rows.tobytes()[:-1].decode("ascii")
    text = ""
    for row in grid:
        for cell in row:
            text += str(int(cell))
        text += "\n"
    return text.strip()

def _tokenize_threads(tokenizer, texts):
    # Training workers inherit TOKENIZERS_PARALLELISM=false from unsloth_zoo, so
    # one batched call encodes on a single core. Fast tokenizers release the GIL
    # while encoding, so Python threads recover the parallelism instead.
    if not getattr(tokenizer, "is_fast", False) or len(texts) < 2:
        return 1
    parallelism = os.environ.get("TOKENIZERS_PARALLELISM")
    if parallelism is None or parallelism.strip().lower() not in ("", "off", "false", "f", "no", "n", "0"):
        return 1
    if sum(len(text) for text in texts) < 65536:
        return 1
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    return max(1, min(int(os.getenv("ARC_TOKENIZE_THREADS", "8")), cpus, len(texts) - 1))


def token_lengths(tokenizer, texts):
    """Return ``len(tokenizer.encode(text))`` for every text."""
    texts = list(texts)
    if not texts:
        return []
    if not getattr(tokenizer, "is_fast", False):
        return [len(tokenizer.encode(text)) for text in texts]
    threads = _tokenize_threads(tokenizer, texts)
    if threads <= 1:
        return [len(ids) for ids in tokenizer(texts)["input_ids"]]
    # The one-row call puts the backend into plain (no truncation or padding)
    # mode; the concurrent calls below then only read that state.
    lengths = [len(tokenizer(texts[:1])["input_ids"][0])]
    rest = texts[1:]
    size = -(-len(rest) // threads)
    chunks = [rest[i:i + size] for i in range(0, len(rest), size)]
    with ThreadPoolExecutor(len(chunks)) as pool:
        for ids in pool.map(lambda chunk: tokenizer(chunk)["input_ids"], chunks):
            lengths.extend(len(row) for row in ids)
    return lengths


def is_valid_solution(guess):
    return isinstance(guess, np.ndarray) and guess.ndim == 2 and all(0 < x <= 30 for x in guess.shape)


def infer_demonstrated_output_shapes(query):
    """Infer test-output shapes from rules shared by every demonstration.

    Prefer a constant input-to-output shape offset because it can preserve
    differences between multiple test-input sizes.  If that rule is absent or
    produces an invalid ARC dimension, fall back to a constant demonstrated
    output shape.  Return one shape per test input, or ``None`` when neither
    rule is supported by all demonstrations.
    """
    train = query.get("train", [])
    test = query.get("test", [])
    if not train or not test:
        return None

    def grid_shape(grid):
        try:
            shape = tuple(int(size) for size in np.shape(grid))
        except (TypeError, ValueError):
            return None
        return shape if len(shape) == 2 else None

    train_shapes = [
        (grid_shape(example.get("input")), grid_shape(example.get("output")))
        for example in train
    ]
    test_shapes = [grid_shape(example.get("input")) for example in test]
    if any(
        input_shape is None or output_shape is None
        for input_shape, output_shape in train_shapes
    ) or any(shape is None for shape in test_shapes):
        return None

    offsets = {
        tuple(
            output_size - input_size
            for input_size, output_size in zip(input_shape, output_shape)
        )
        for input_shape, output_shape in train_shapes
    }
    if len(offsets) == 1:
        offset = next(iter(offsets))
        inferred = [
            tuple(size + delta for size, delta in zip(shape, offset))
            for shape in test_shapes
        ]
        if all(all(0 < size <= 30 for size in shape) for shape in inferred):
            return inferred

    output_shapes = {output_shape for _, output_shape in train_shapes}
    if len(output_shapes) == 1:
        output_shape = next(iter(output_shapes))
        if all(0 < size <= 30 for size in output_shape):
            return [output_shape] * len(test_shapes)
    return None


def infer_demonstrated_output_shape(query):
    """Return the single test-output shape inferred by conservative rules."""
    inferred = infer_demonstrated_output_shapes(query)
    if inferred is None or len(inferred) != 1:
        return None
    return inferred[0]


def shuffled(data_list):
    return np.random.permutation(data_list).tolist()

def permute_mod(a, descriptor, invert=False):
    permutation = [int(i) for i in descriptor if str(i).isdigit()]
    assert sorted(permutation)==list(range(10))
    a = np.asarray(a)
    if a.ndim==3:
        if not invert: permutation = np.argsort(permutation)
        a = a[..., permutation]
    else:
        assert a.ndim==2
        if invert: permutation = np.argsort(permutation)
        a = np.asarray(permutation)[a]
    return a

def permute_rnd_all_(query):
    permutation = np.random.permutation(10).tolist()
    return 'permute' + ''.join(map(str, permutation))


def query_colors(query):
    colors = set()
    for examples in query.values():
        for example in examples:
            for grid in example.values():
                colors.update(np.asarray(grid).reshape(-1).tolist())
    return sorted(int(color) for color in colors)


def balanced_color_descriptors(query, n, seed, key):
    """Build deterministic mappings that are unique on observed colors."""
    used_colors = query_colors(query)
    if not used_colors or n <= 0:
        return []

    seed_text = f"{seed}:{key}"
    local_seed = int.from_bytes(
        hashlib.sha256(seed_text.encode("utf-8")).digest()[:8],
        byteorder="big",
    )
    rng = np.random.default_rng(local_seed)
    seen_effective_mappings = set()
    descriptors = []

    total_capacity = math.perm(10, len(used_colors))
    if 0 in used_colors:
        fixed_capacity = math.perm(9, len(used_colors) - 1)
    else:
        fixed_capacity = 0
    fixed_count = 0

    def sample_descriptor(keep_zero):
        nonlocal fixed_count
        if len(seen_effective_mappings) >= total_capacity:
            return None
        if keep_zero and fixed_count >= fixed_capacity:
            return None

        for _ in range(4096):
            permutation = rng.permutation(10)
            if keep_zero:
                zero_position = int(np.where(permutation == 0)[0][0])
                permutation[0], permutation[zero_position] = (
                    permutation[zero_position],
                    permutation[0],
                )
            elif keep_zero is False and permutation[0] == 0:
                permutation[0], permutation[1] = permutation[1], permutation[0]

            effective_mapping = tuple(
                int(permutation[color]) for color in used_colors
            )
            if effective_mapping in seen_effective_mappings:
                continue

            seen_effective_mappings.add(effective_mapping)
            if keep_zero:
                fixed_count += 1
            return "permute" + "".join(map(str, permutation.tolist()))
        return None

    for i in range(n):
        keep_zero = (i % 2 == 0) if 0 in used_colors else None
        descriptor = sample_descriptor(keep_zero)
        if descriptor is None and keep_zero is not None:
            descriptor = sample_descriptor(not keep_zero)
        if descriptor is None:
            # A one-color puzzle has only ten distinct effective mappings.
            # Reuse them only after the useful mapping space is exhausted.
            descriptor = descriptors[i % len(descriptors)]
        descriptors.append(descriptor)
    return descriptors


class QwenFormatter:

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def fmt_query(self, query) -> str:
        grid_input = convert_grid_to_string(query[0]["input"])
        return "<|im_start|>user\n" + grid_input + "<|im_end|><|im_start|>assistant\n"

    def fmt_reply(self, reply) -> str:
        return convert_grid_to_string(reply[0]) + "<|im_end|>"

    def fmt_train(self, train, last_is_challenge=False) -> str:
        if last_is_challenge:
            test = train[-1]
            train = train[:-1]
        else:
            test = None
        text = ""
        for x in train:
            grid_input = convert_grid_to_string(x["input"])
            grid_output = convert_grid_to_string(x["output"])
            text += f"<|im_start|>user\n{grid_input}<|im_end|><|im_start|>assistant\n{grid_output}<|im_end|>"
        if test is not None:
            text += self.fmt_query([test]) + self.fmt_reply([test["output"]])
        return text

    def max_new_tokens(self):
        max_sized_reply = np.zeros([30, 30], dtype=int)
        tokens = self.tokenizer.encode(self.fmt_reply([max_sized_reply]))
        return len(tokens) + 1

    def convert_tokens_to_array(self, tokens, limit_rows=30):
        if len(tokens) < 2:
            return None
        text = self.tokenizer.decode(tokens[:-1])
        try:
            lines = text.strip().split("\n")
            by_rows = [row for row in [[int(x) for x in line if x.isdigit()] for line in lines] if len(row)]
            if len(by_rows) > limit_rows:
                by_rows = by_rows[:limit_rows]
            array = np.array(by_rows, dtype=int)
            if is_valid_solution(array):
                return array
        except:
            pass
        return None


class ArcDataset:

    @staticmethod
    def forward_mod(a, key, use_perm=True):
        if a is None: return a
        for op in key.split('.')[1:]:
            if   op=='rot90':              a = np.rot90(a)
            elif op=='transpose':          a = np.swapaxes(a, 0, 1)
            elif op.startswith('permute'): a = permute_mod(a, op, invert=False) if use_perm else a
            elif op.startswith('copy'):    a = np.copy(a)
            elif op.startswith('out'):     a = a
            elif op.startswith('ex'):      a = a
            elif op.startswith('run'):     a = a
            else: raise NotImplementedError(f"Inversion of operation '{op}' unknown.")
        return a

    @staticmethod
    def invert_mod(a, key, inv_perm=True):
        if a is None: return a
        for op in key.split('.')[1:][::-1]:
            if   op=='rot90':              a = np.rot90(a, k=3)
            elif op=='transpose':          a = np.swapaxes(a, 0, 1)
            elif op.startswith('permute'): a = permute_mod(a, op, invert=True) if inv_perm else a
            elif op.startswith('copy'):    a = np.copy(a)
            elif op.startswith('out'):     a = a
            elif op.startswith('ex'):      a = a
            elif op.startswith('run'):     a = a
            else: raise NotImplementedError(f"Inversion of operation '{op}' unknown.")
        return a

    def __init__(self, queries, replies={}, keys=None, is_orig=False):
        if keys is not None: keys = [k for k in keys if k is not None]
        self.queries = queries if keys is None else {k: queries[k] for k in keys}
        self.replies = replies if keys is None else {k: replies[k] for k in keys if k in replies}
        self.is_orig = is_orig
        self.keys = sorted(queries.keys()) if keys is None else keys
        self.transposed_dataset = None

    def change_keys(self, keys, keep_flags=False):
        flags = dict(is_orig=self.is_orig) if keep_flags else {}
        return self.__class__(queries=self.queries, replies=self.replies, keys=keys, **flags)

    @classmethod
    def from_file(cls, queries_file, keys=None):
        with open(queries_file) as f:
            queries = f.read()
        return cls(
            queries=json.loads(queries),
            is_orig=True,
            keys=keys,
        )

    def load_replies(self, replies_file):
        print(f"*** Load solutions from '{replies_file}'...")
        with open(replies_file) as f: replies = f.read()
        replies_parsed = json.loads(replies)
        self.replies = {k: replies_parsed[k] for k in self.keys}
        return self

    def split_multi_replies(self):
        key_indices = [(k, i) for k in self.keys for i in range(len(self.queries[k]['test']))]
        return self.__class__(
            keys=[f'{k}_{i}' for k, i in key_indices],
            queries={f'{k}_{i}': {'train': self.queries[k]['train'], 'test': [self.queries[k]['test'][i]]} for k, i in key_indices},
            replies={f'{k}_{i}': [self.replies[k][i]] for k, i in key_indices if k in self.replies},
        )

    def split_train_replies(self):
        """Expose every demonstration as a leave-one-out pseudo-test.

        The held-out pair's input becomes the query and its output becomes the
        known reply.  All remaining demonstrations stay in the prompt.  This
        mirrors the completion objective used by task-time training without
        leaking the target output into its inference prompt.
        """
        keys = []
        queries = {}
        replies = {}
        for task_key in self.keys:
            train_examples = self.queries[task_key].get("train", [])
            for train_index, target in enumerate(train_examples):
                key = train_prediction_key(task_key, train_index)
                keys.append(key)
                queries[key] = {
                    "train": [
                        example
                        for index, example in enumerate(train_examples)
                        if index != train_index
                    ],
                    "test": [{"input": target["input"]}],
                }
                replies[key] = [target["output"]]
        return self.__class__(
            queries=queries,
            replies=replies,
            keys=keys,
        )

    def shuffled(self):
        return self.__class__(queries=self.queries, replies=self.replies, keys=shuffled(self.keys))

    def append(*datasets):
        return datasets[0].__class__(
            queries={k: v for d in datasets for k, v in d.queries.items()},
            replies={k: v for d in datasets for k, v in d.replies.items()},
            keys   =[k    for d in datasets for k    in d.keys           ],
        )

    def mod_single(self, mod_func, descriptor, i, keep_key, inputs_only):
        queries = {}
        replies = {}
        keys    = []
        for k0 in self.keys:
            if descriptor is None:
                desc = ('copy{i}' if mod_func is np.copy else mod_func.__name__).format(i=i)
            elif isinstance(descriptor, str):
                desc = descriptor.format(i=i)
            elif isinstance(descriptor, dict):
                desc = descriptor[k0][i]
            else:
                desc = descriptor(self.queries[k0]).format(i=i)
            func = lambda a, d: np.asarray(mod_func(a) if descriptor is None else mod_func(a, d)).tolist()
            k1 = k0 if keep_key else f"{k0}.{'I' if inputs_only else ''}{desc}"
            keys.append(k1)
            queries[k1] = {m: [{t: (func(a, desc) if t=='input' or not inputs_only else a) for t, a in x.items()} for x in e] for m, e in self.queries[k0].items()}
            if k0 in self.replies:
                replies[k1] = [func(a, desc) for a in self.replies[k0]]
        ret = self.__class__(queries=queries, replies=replies, keys=keys)
        return ret

    def mod(self, mod_func, descriptor=None, n=1, stack=None, keep=False, keep_key=False, shuffle=False, join=True, inputs_only=False):
        assert not (keep and keep_key)
        cur = self
        ret = [cur.shuffled() if shuffle else cur] if keep else []
        if stack is None: stack = mod_func.__name__.startswith('rot')
        for i in range(n):
            cur = (cur if stack else self).mod_single(mod_func, descriptor, i=i, keep_key=keep_key, inputs_only=inputs_only)
            ret.append(cur.shuffled() if shuffle else cur)
        return self.__class__.append(*ret) if join else ret

    def get(self, key, formatter: QwenFormatter):
        train = formatter.fmt_train(self.queries[key]['train'])
        query = formatter.fmt_query(self.queries[key]['test'])
        reply = formatter.fmt_reply(self.replies[key]) if key in self.replies else ''
        text = train+query+reply if reply else formatter.fmt_train(self.queries[key]['train'], last_is_challenge=True)
        return dict(key=key, train=train, query=query, reply=reply, input=train+query, text=text)

    def as_list(self, formatter: QwenFormatter):
        return [self.get(key, formatter) for key in self.keys]

    def get_length(self, key, formatter: QwenFormatter, name, max_of_transposed=False):
        if formatter is None:
            if   name=='input': return sum(np.prod(np.shape(v)) for v3 in self.queries[key].values() for v2 in v3 for v in v2.values())
            elif name=='reply': return sum(np.prod(np.shape(v)) for v in self.replies[key])
            else: assert False
        else:
            datasets = [self]
            if max_of_transposed:
                if self.transposed_dataset is None: self.transposed_dataset = self.mod(np.transpose, keep=False, keep_key=True)
                datasets.append(self.transposed_dataset)
            return max(len(formatter.tokenizer.encode(ds.get(key, formatter=formatter)[name])) for ds in datasets)

    def cut_to_len(self, formatter, name, max_len, from_end=False):
        temp_ds = self.change_keys(self.keys)
        new_keys = []
        new_queries = {}
        new_replies = {}
        if formatter is None:
            initial_lengths = [temp_ds.get_length(key, formatter=None, name=name) for key in self.keys]
        else:
            # Tokenize every row once in a single batch; only rows that are too
            # long fall back to the per-row loop below.
            initial_lengths = token_lengths(
                formatter.tokenizer,
                (temp_ds.get(key, formatter=formatter)[name] for key in self.keys),
            )
        truncated = False
        for key, length in zip(self.keys, initial_lengths):
            if truncated:
                # Truncation adds keys to the shared query dict, so later rows
                # are measured one by one exactly as before.
                length = temp_ds.get_length(key, formatter=formatter, name=name)
            reply = temp_ds.replies.get(key)
            while max_len<length:
                truncated = True
                query = temp_ds.queries[key]
                if not query['train']:
                    raise ValueError(
                        f"Cannot fit dataset row {key!r}: minimum {name} "
                        f"length={length} with no training "
                        f"demonstrations left exceeds max_len={max_len}"
                    )
                if not key.split('.')[-1].startswith('ex'):
                    key = f"{key}.ex{''.join(map(str, range(len(query['train']))))}"
                key_split = key.split('.')
                assert key_split[-1].startswith('ex')
                key = '.'.join(key_split[:-1] + [f'ex{key_split[-1][2:-1] if from_end else key_split[-1][3:]}'])
                temp_ds.queries[key] = {k: ((v[:-1] if from_end else v[1:]) if k=='train' else v) for k, v in query.items()}
                if reply is not None:
                    temp_ds.replies[key] = reply
                length = temp_ds.get_length(key, formatter=formatter, name=name)
            new_keys.append(key)
            new_queries[key] = temp_ds.queries[key]
            if reply is not None: new_replies[key] = reply
        return self.__class__(keys=new_keys, queries=new_queries, replies=new_replies)

    def shuffle_ex(self, perm=None, keep_max=None):
        new_keys = []
        new_queries = {}
        new_replies = {}
        for key in self.keys:
            n = len(self.queries[key]['train'])
            p = np.random.permutation(n) if perm is None else perm
            if keep_max is not None: p = p[:keep_max]
            new_key = f'{key}.ex' + ('-' if (p.max()>9) else '').join(map(str, p.tolist()))
            new_keys.append(new_key)
            new_queries[new_key] = {k: (np.array(v, dtype=object)[p].tolist() if k=='train' else v) for k, v in self.queries[key].items()}
            if key in self.replies: new_replies[new_key] = self.replies[key]
        return self.__class__(queries=new_queries, replies=new_replies, keys=new_keys)

    def balanced_shuffle_ex(self):
        """Balance every training pair across prompt positions deterministically."""
        new_keys = []
        new_queries = {}
        new_replies = {}
        for sample_index, key in enumerate(self.keys):
            n = len(self.queries[key]['train'])
            if n == 0:
                p = np.array([], dtype=int)
            else:
                p = np.arange(n)
                block, shift = divmod(sample_index, n)
                if block % 2:
                    p = p[::-1]
                p = np.roll(p, -shift)
            separator = '-' if (len(p) and p.max() > 9) else ''
            new_key = f'{key}.ex' + separator.join(map(str, p.tolist()))
            new_keys.append(new_key)
            new_queries[new_key] = {
                split: (np.array(examples, dtype=object)[p].tolist() if split == 'train' else examples)
                for split, examples in self.queries[key].items()
            }
            if key in self.replies:
                new_replies[new_key] = self.replies[key]
        return self.__class__(queries=new_queries, replies=new_replies, keys=new_keys)

    def train_augmentation_count(self, default_n=16, small_n=32, max_train_pairs=2, max_average_grid_area=100):
        """Use more color variants for tasks with little supervised content."""
        for query in self.queries.values():
            train = query.get('train', [])
            if len(train) <= max_train_pairs:
                return small_n
            grid_areas = [
                int(np.prod(np.shape(grid)))
                for example in train
                for grid in example.values()
            ]
            if grid_areas and float(np.mean(grid_areas)) < max_average_grid_area:
                return small_n
        return default_n

    def with_min_train_pairs(self, min_train_pairs=0, extra_train_pair_mode="augment", seed=0):
        """Expand each conversation without changing row count or target replies."""
        from arc_train_pair_expansion import expand_training_query, validate_pair_expansion

        validate_pair_expansion(min_train_pairs, extra_train_pair_mode)
        if min_train_pairs == 0:
            return self
        queries, records = {}, {}
        for key in self.keys:
            queries[key], records[key] = expand_training_query(
                self.queries[key], key=key, min_train_pairs=min_train_pairs,
                mode=extra_train_pair_mode, seed=seed,
            )
        result = self.__class__(queries=queries, replies=self.replies, keys=self.keys)
        result.pair_expansion_records = records
        return result

    def augment(self, n=1, shfl_keys=False, seed=42, unique_colors=False, balanced_examples=False, min_train_pairs=0, extra_train_pair_mode="augment"):
        np.random.seed(seed)
        d = self
        d = d.mod(np.transpose, keep=True)
        d = d.mod(np.rot90, n=3, keep=True)
        if unique_colors:
            descriptors = {
                key: balanced_color_descriptors(d.queries[key], n, seed, key)
                for key in d.keys
            }
            d = d.mod(permute_mod, descriptors, n=n, keep=False)
        else:
            d = d.mod(permute_mod, permute_rnd_all_, n=n, shuffle=shfl_keys, keep=False)
        d = d.balanced_shuffle_ex() if balanced_examples else d.shuffle_ex()
        if unique_colors and shfl_keys:
            d = d.shuffled()
        if min_train_pairs:
            d = d.with_min_train_pairs(min_train_pairs, extra_train_pair_mode, seed=seed)
        return d

    def augment_exact_views(self, count, seed=42, min_train_pairs=0, extra_train_pair_mode="augment"):
        """Return exactly ``count`` deterministic views of one dataset key.

        ``augment(n)`` naturally creates views in groups of eight (four
        rotations, with and without transpose). Rescoring is performed one
        candidate key at a time, so generate enough complete groups and keep
        the deterministic prefix when a caller requests another positive
        count such as 10 or 24.
        """
        count = int(count)
        if count <= 0:
            raise ValueError("view count must be positive")
        if len(self.keys) != 1:
            raise ValueError(
                "augment_exact_views expects exactly one source key, got "
                f"{len(self.keys)}"
            )
        augmented = self.augment(n=math.ceil(count / 8), seed=seed,
                                 min_train_pairs=min_train_pairs,
                                 extra_train_pair_mode=extra_train_pair_mode)
        return augmented.change_keys(augmented.keys[:count])

    def get_submission(self, results=None):
        assert self.is_orig==True, 'Must be run on original dataset.'
        submission = {k: [{f'attempt_{i+1}': [[0]] for i in range(2)} for _ in range(len(self.queries[k]['test']))] for k in self.keys}
        if results is not None: self.fill_submission(results, submission)
        return submission

    @staticmethod
    def fill_submission(results, submission):
        print(f'*** Generating submission for {len(results)} outputs...')
        for k, v in results.items():
            base_id, base_nr = k.split('_')
            target_dict = submission[base_id][int(base_nr)]
            guesses = list(v[:len(target_dict)])
            if guesses and len(guesses) < len(target_dict):
                guesses.extend(
                    [guesses[-1]] * (len(target_dict) - len(guesses))
                )
            for i, g in enumerate(guesses):
                target_dict[f'attempt_{i+1}'] = g.tolist()

    def validate_submission(self, submission):
        assert self.is_orig==True, 'Must be run on original dataset.'
        score = 0
        for k, v in self.replies.items():
            for i, r in enumerate(v):
                for attempt in ['attempt_1', 'attempt_2']:
                    if np.array_equal(r, submission[k][i][attempt]):
                        score += 1 / len(v)
                        break
        return score
