"""Streams files of a (private) Hugging Face repo directly into CNS, no local disk.

Resumable: files whose CNS size already matches the Hub size are skipped.

Example:
  blaze run //experimental/users/arazdai/kaggle/sft:download_hf_to_cns -- \
    --repo_id=chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-img-E128-2bit-causal \
    --repo_type=model \
    --cns_out_dir=/cns/lu-d/home/arazdai/kaggle/models/dsv41_img_e128_2bit_causal
"""

from __future__ import annotations

from concurrent import futures
import json
import os
import re
import time
from typing import Any

from absl import app
from absl import flags
import huggingface_hub
import requests

from google3.experimental.users.arazdai.kaggle.sft import export_and_quantize
from google3.pyglib import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string("repo_id", None, "Hugging Face repo id.", required=True)
flags.DEFINE_enum("repo_type", "model", ["model", "dataset"], "Hub repo type.")
flags.DEFINE_string("revision", "main", "Git revision.")
flags.DEFINE_string("cns_out_dir", None, "Target CNS directory.", required=True)
flags.DEFINE_string(
    "include_regex", ".*", "Only download repo paths matching this regex."
)
flags.DEFINE_string("exclude_regex", "", "Skip repo paths matching this regex.")
flags.DEFINE_string(
    "token_path",
    "/usr/local/google/home/arazdai/.cache/huggingface/token",
    "File containing the HF access token (never printed).",
)
flags.DEFINE_integer("num_workers", 8, "Parallel streams.")
flags.DEFINE_bool("dry_run", False, "List the plan without downloading.")


def _list_files(api: huggingface_hub.HfApi) -> list[tuple[str, int]]:
  """Returns (path, size) for every file in the repo (recursive)."""
  out: list[tuple[str, int]] = []
  for entry in api.list_repo_tree(
      FLAGS.repo_id,
      repo_type=FLAGS.repo_type,
      revision=FLAGS.revision,
      recursive=True,
  ):
    size = getattr(entry, "size", None)
    if size is None:  # directories
      continue
    out.append((entry.path, int(size)))
  return out


def _cns_size(path: str) -> int:
  if not gfile.Exists(path):
    return -1
  try:
    return int(gfile.Stat(path).length)
  except Exception:  # pylint: disable=broad-exception-caught
    return -1


def _stream_one(rel: str, size: int, token: str) -> dict[str, Any]:
  dst = os.path.join(FLAGS.cns_out_dir, rel)
  if _cns_size(dst) == size:
    return {"file": rel, "status": "already_exists", "bytes": size}
  url = huggingface_hub.hf_hub_url(
      FLAGS.repo_id, rel, repo_type=FLAGS.repo_type, revision=FLAGS.revision
  )
  gfile.MakeDirs(os.path.dirname(dst))
  t0 = time.time()
  n = 0
  # `requests` drops the Authorization header on cross-host redirects (CDN).
  with requests.get(
      url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=300
  ) as resp:
    resp.raise_for_status()
    with gfile.Open(dst + ".partial", "wb") as f_out:
      for chunk in resp.iter_content(chunk_size=16 << 20):
        if chunk:
          f_out.write(chunk)
          n += len(chunk)
  if size and n != size:
    raise IOError(f"{rel}: downloaded {n} bytes, expected {size}")
  gfile.Rename(dst + ".partial", dst, overwrite=True)
  dt = max(1e-3, time.time() - t0)
  return {
      "file": rel,
      "status": "downloaded",
      "bytes": n,
      "mb_per_sec": round(n / 1e6 / dt, 1),
      "elapsed_sec": round(dt, 1),
  }


def main(argv: list[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  with open(FLAGS.token_path) as f:
    token = f.read().strip()
  api = huggingface_hub.HfApi(token=token)
  inc = re.compile(FLAGS.include_regex)
  exc = re.compile(FLAGS.exclude_regex) if FLAGS.exclude_regex else None
  files = [
      (p, s)
      for p, s in _list_files(api)
      if inc.search(p) and not (exc and exc.search(p))
  ]
  total = sum(s for _, s in files)
  print(
      f"{FLAGS.repo_type} {FLAGS.repo_id}@{FLAGS.revision}: {len(files)} files, "
      f"{total / 1e9:.2f} GB -> {FLAGS.cns_out_dir}",
      flush=True,
  )
  todo = [(p, s) for p, s in files if _cns_size(os.path.join(FLAGS.cns_out_dir, p)) != s]
  print(f"  {len(files) - len(todo)} already present, {len(todo)} to download "
        f"({sum(s for _, s in todo) / 1e9:.2f} GB)", flush=True)
  if FLAGS.dry_run:
    for p, s in todo:
      print(f"    {s / 1e6:10.1f} MB  {p}")
    return
  export_and_quantize.ensure_cns_quota_dir(FLAGS.cns_out_dir)
  # Largest files first so the tail of the run is short.
  todo.sort(key=lambda ps: -ps[1])
  t_start = time.time()
  done_bytes = 0
  with futures.ThreadPoolExecutor(max_workers=FLAGS.num_workers) as pool:
    futs = {pool.submit(_stream_one, p, s, token): p for p, s in todo}
    for i, fut in enumerate(futures.as_completed(futs), 1):
      res = fut.result()
      done_bytes += res["bytes"]
      print(f"  [{i}/{len(todo)}] {json.dumps(res)}", flush=True)
  dt = max(1e-3, time.time() - t_start)
  print(
      f"Done: {done_bytes / 1e9:.2f} GB in {dt / 60:.1f} min "
      f"({done_bytes / 1e6 / dt:.1f} MB/s aggregate).",
      flush=True,
  )
  with gfile.Open(os.path.join(FLAGS.cns_out_dir, "HF_DOWNLOAD_MANIFEST.json"), "w") as f:
    json.dump(
        {
            "repo_id": FLAGS.repo_id,
            "repo_type": FLAGS.repo_type,
            "revision": FLAGS.revision,
            "files": [{"path": p, "size": s} for p, s in files],
            "downloaded_at": time.time(),
        },
        f,
        indent=2,
    )


if __name__ == "__main__":
  app.run(main)
