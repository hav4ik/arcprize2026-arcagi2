#!/usr/bin/env python3
"""Dump API name vocabularies (builtins, keywords, stdlib modules, numpy/scipy attributes, common types' methods)
to out/api_names.json. Run with a Python that has scipy (e.g. miniconda python3); the .venv lacks scipy."""
import builtins, collections, itertools, json, keyword, sys, os
import numpy as np
import scipy, scipy.ndimage, scipy.signal, scipy.spatial, scipy.sparse, scipy.optimize, scipy.stats, scipy.special
import scipy.sparse.csgraph

out = {}
out["builtins"] = sorted(dir(builtins))
out["keywords"] = sorted(keyword.kwlist + list(getattr(keyword, "softkwlist", [])))
out["stdlib_modules"] = sorted(sys.stdlib_module_names)
mods = {"numpy": np, "np.linalg": np.linalg, "np.random": np.random, "scipy": scipy, "scipy.ndimage": scipy.ndimage,
        "scipy.signal": scipy.signal, "scipy.spatial": scipy.spatial, "scipy.sparse": scipy.sparse,
        "scipy.optimize": scipy.optimize, "scipy.stats": scipy.stats, "scipy.special": scipy.special,
        "scipy.sparse.csgraph": scipy.sparse.csgraph, "collections": collections, "itertools": itertools}
import functools, math, copy, re, operator, heapq, random, string
for m in (functools, math, copy, re, operator, heapq, random, string, json, os):
    mods[m.__name__] = m
out["module_attrs"] = {k: sorted(n for n in dir(v) if not n.startswith("__")) for k, v in mods.items()}
types = {"ndarray": np.ndarray, "list": list, "dict": dict, "set": set, "frozenset": frozenset, "tuple": tuple,
         "str": str, "int": int, "deque": collections.deque, "Counter": collections.Counter,
         "defaultdict": collections.defaultdict, "generic": np.generic}
out["type_attrs"] = {k: sorted(n for n in dir(v) if not n.startswith("__")) for k, v in types.items()}
out["third_party_modules"] = ["numpy", "scipy", "skimage", "sklearn", "networkx", "pandas", "PIL", "cv2", "sympy"]
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "api_names.json"), "w"))
print({k: len(v) for k, v in out.items()})
