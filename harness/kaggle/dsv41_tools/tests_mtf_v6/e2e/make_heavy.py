"""Synthetic heavy challenge file: multi-test tasks (4-6 test inputs), 30x30 grids, 10 train pairs, odd task ids."""
import json, random, sys
rnd = random.Random(7)
def grid(h, w):
    return [[rnd.randint(0, 9) for _ in range(w)] for _ in range(h)]
ids = ["0", "00000000", "ffffffff", "Task-With-Dash_and_Underscore", "日本語タスク", "a.b", "with space", "x" * 64,
       "a/b", "..", "-1", "1e5", "None", "true", "{}", "tab\there", "ünïcödé", "UPPER", "a:b", "#hash"]
ids += [f"heavy{i:02d}" for i in range(40 - len(ids))]
tasks = {}
for i, t in enumerate(ids):
    n_test = [4, 5, 6][i % 3]
    same = i % 2 == 0  # same-shape test inputs (3-D numpy path) vs mixed shapes
    shapes = [(30, 30)] * n_test if same else [(30, 30), (1, 30), (30, 1), (17, 23), (1, 1), (29, 30)][:n_test]
    train = []
    for j in range(10):
        g = grid(30, 30)
        train.append({"input": g, "output": [r[::-1] for r in g] if j % 2 else [list(r) for r in g[::-1]]})
    tasks[t] = {"train": train, "test": [{"input": grid(h, w)} for h, w in shapes]}
for i in range(10):  # single-test heavy tasks
    train = [{"input": grid(30, 30), "output": grid(30, 30)} for _ in range(10)]
    tasks[f"single{i}"] = {"train": train, "test": [{"input": grid(30, 30)}]}
json.dump(tasks, open(sys.argv[1], "w"))
print(len(tasks), "tasks,", sum(len(v["test"]) for v in tasks.values()), "pairs,",
      len(json.dumps(tasks)) // 1024, "KiB")
