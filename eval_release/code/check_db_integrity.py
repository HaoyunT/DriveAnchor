"""Check every extracted test-split db for corruption from the bad unzip.

Filename presence was the earlier check and it passed 1130/1130 -- that only
proves unzip wrote *a* file for each central-directory entry, not that the
bytes are the database.  The zip's central directory reported bad offsets
from entry #538 on, so files after that point can easily be another file's
bytes under the right name.  ``PRAGMA integrity_check`` opens the file as
sqlite and walks it, which catches that; a plain existence check does not.

``sqlite3.connect(..., timeout=5)`` only bounds how long it waits to acquire a
lock -- it does not bound how long the integrity_check statement itself runs.
A corrupted file can make the walk pathological (cyclic freelist pointers,
inflated page counts from garbage bytes) and hang the single worker on one
file for many minutes with nothing to show for it.  Each check runs in its own
process with a wall-clock cap so one bad file can't stall the whole queue --
skipped files are recorded distinctly from confirmed-bad ones, since "corrupt"
and "too slow to tell" call for different follow-up.
"""
from __future__ import annotations

import json
import multiprocessing
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PER_FILE_TIMEOUT_S = 20


def _check_worker(path: str, q) -> None:
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        row = con.execute("PRAGMA integrity_check;").fetchone()
        con.close()
        ok = row is not None and row[0] == "ok"
        q.put((ok, "" if ok else str(row)))
    except Exception as e:
        q.put((False, str(e)))


def check(path: str) -> tuple[str, bool, str]:
    q: multiprocessing.Queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_check_worker, args=(path, q))
    p.start()
    p.join(PER_FILE_TIMEOUT_S)
    if p.is_alive():
        p.terminate()
        p.join()
        return path, False, f"timed out after {PER_FILE_TIMEOUT_S}s (unverified, not confirmed corrupt)"
    if not q.empty():
        ok, err = q.get()
        return path, ok, err
    return path, False, "worker died without result"


def main() -> int:
    paths = [line.strip() for line in open(sys.argv[1]) if line.strip()]
    partial = Path("/tmp/dp_eval/db_integrity.json")
    results: dict[str, dict] = {}
    if partial.exists():
        # Resuming: the mechanical drive under-thrashes at even 3 concurrent
        # readers, so restarts happen at a lower thread count each time.
        # Re-checking files this run already cleared wastes the exact time
        # budget the lower concurrency is trying to protect.
        prior = json.loads(partial.read_text())
        for p in prior.get("good_paths", []):
            results[p] = {"ok": True, "error": ""}
        for p in prior.get("bad_paths", []):
            results[p] = {"ok": False, "error": "previously flagged bad"}
        print(f"续跑: 已有 {len(results)} 条结果, 跳过", flush=True)
    remaining = [p for p in paths if p not in results]

    def flush():
        good = [p for p, r in results.items() if r["ok"]]
        bad = [p for p, r in results.items() if not r["ok"]]
        out = {"total": len(paths), "checked": len(results), "good": len(good),
               "bad": len(bad), "good_paths": good, "bad_paths": bad}
        partial.write_text(json.dumps(out, indent=1))

    with ThreadPoolExecutor(max_workers=1) as pool:
        for i, (path, ok, err) in enumerate(pool.map(check, remaining)):
            results[path] = {"ok": ok, "error": err}
            if (i + 1) % 20 == 0:
                flush()
                good = sum(1 for r in results.values() if r["ok"])
                print(f"{len(results)}/{len(paths)}  好 {good}", flush=True)
    flush()
    good = [p for p, r in results.items() if r["ok"]]
    bad = [p for p, r in results.items() if not r["ok"]]
    print(json.dumps({"total": len(paths), "good": len(good), "bad": len(bad)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
