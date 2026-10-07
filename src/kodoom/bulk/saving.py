"""Periodic verified saves, retries and bounded unsaved work (BULK-06)."""

from __future__ import annotations

import threading
import time

from kodoom.bulk.remote import WriterConflict


class StopWork(RuntimeError):
    pass


class AutoSaver:
    def __init__(self, campaign, remote, request, *, run=None, clock=time.monotonic):
        self.campaign, self.remote, self.request, self.run = campaign, remote, request, run
        self.clock = clock
        self.last_success = clock()
        self.error = None
        self.paused = False
        self.fenced = False
        self.stop_event = threading.Event()
        self.thread = None

    def sync_once(self):
        snapshot = self.campaign.snapshot()
        if self.run is not None:

            def write(directory):
                from kodoom.bulk.state import atomic_write

                for name, data in snapshot.items():
                    atomic_write(directory / name, data)

            self.run.save_latest(len(self.campaign.progress["completed"]), write)
        revision = self.remote.save(snapshot)
        self.last_success, self.error = self.clock(), None
        self.campaign.event(
            "remote-save-verified",
            revision=revision,
            completed=len(self.campaign.progress["completed"]),
        )
        self.paused = self.remote.paused()
        print(f"Verified remote checkpoint {revision[:12]}; {self.campaign.counts()}", flush=True)
        return revision

    def flush(self):
        for retry in range(3):
            try:
                return self.sync_once()
            except WriterConflict:
                self.fenced = True
                self.error = "writer-conflict"
                raise
            except Exception as exc:
                self.error = type(exc).__name__
                self.campaign.event("remote-save-failed", error_type=self.error, retry=retry)
                if retry == 2:
                    raise
                self.stop_event.wait(2**retry)

    def start(self):
        self.flush()

        def loop():
            while not self.stop_event.wait(self.request["upload_seconds"]):
                try:
                    self.flush()
                except Exception:
                    print(
                        "Remote save failed; local progress retained; retry is scheduled",
                        flush=True,
                    )

        self.thread = threading.Thread(target=loop, name="kodoom-checkpoints", daemon=True)
        self.thread.start()

    def ensure_safe(self):
        if self.fenced:
            raise StopWork("writer-conflict")
        if self.paused:
            raise StopWork("paused-awaiting-review")
        with self.campaign.lock:
            unsaved = any(
                ref["path"] not in self.remote.last_manifest
                for ref in self.campaign.progress["completed"].values()
            )
        if unsaved and self.clock() - self.last_success >= self.request["max_unsaved_seconds"]:
            raise StopWork("remote-save-overdue")

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=60)
            if self.thread.is_alive():
                raise StopWork("remote-upload-still-running")
