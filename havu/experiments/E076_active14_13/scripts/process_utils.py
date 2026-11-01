"""Cleanup captured process identities; tolerate Vast hosts without usable pidfd."""
import time
import psutil


def terminate_owned(processes, grace=15):
    for proc in processes:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    deadline = time.monotonic() + grace
    alive = list(processes)
    while alive and time.monotonic() < deadline:
        pending = []
        for proc in alive:
            try:
                if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                    pending.append(proc)
            except psutil.NoSuchProcess:
                pass
        alive = pending
        if alive:
            time.sleep(.25)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
