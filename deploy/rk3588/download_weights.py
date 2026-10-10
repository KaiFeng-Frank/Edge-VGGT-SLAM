#!/usr/bin/env python3
"""Resume the official VGGT checkpoint on external storage using verified ranges."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
import uuid


MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
MODEL_SIZE = 5026874952
MODEL_SHA256 = "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0"
STOP_REQUESTED = threading.Event()


def install_process_dns_cache():
    """Reduce resolver traffic without changing system DNS, hostnames, or TLS."""
    original = socket.getaddrinfo
    resolved = {}
    positions = {}
    lock = threading.RLock()

    def cached(host, port, family=0, type=0, proto=0, flags=0):
        key = (host, port, family, type, proto, flags)
        with lock:
            if key not in resolved:
                for attempt in range(3):
                    try:
                        resolved[key] = original(host, port, family, type, proto, flags)
                        break
                    except socket.gaierror:
                        if attempt == 2:
                            raise
                        time.sleep(0.5 * (attempt + 1))
            entries = resolved[key]
            position = positions.get(key, 0) % len(entries)
            positions[key] = position + 1
            return entries[position:] + entries[:position]

    socket.getaddrinfo = cached


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def unique_url(url, start, attempt):
    # Preserve every byte of a signed CDN URL; re-encoding its existing query
    # can invalidate the signature. The unique argument avoids cached ranges.
    if urlsplit(url).hostname not in {"huggingface.co", "hf-mirror.com"}:
        return url
    separator = "&" if "?" in url else "?"
    query = [("download", "true"), ("rk3588_range", f"{start}-{attempt}-{uuid.uuid4().hex}")]
    return url + separator + urlencode(query)


class InvalidRangeResponse(RuntimeError):
    pass


def download(args):
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.stat().st_size == args.size and sha256_file(destination) == args.sha256:
            print(f"Already verified: {destination}", flush=True)
            return
        raise RuntimeError(f"An existing final file failed size/hash verification: {destination}")

    partial = destination.with_name(destination.name + ".part")
    manifest_path = destination.with_name(destination.name + ".ranges.json")
    metadata = {"url": args.url, "size": args.size, "sha256": args.sha256, "chunk_size": args.chunk_size}
    completed = set()
    if manifest_path.is_file():
        saved = json.loads(manifest_path.read_text())
        if any(saved.get(key) != metadata[key] for key in ("url", "size", "sha256")):
            raise RuntimeError("Resume metadata differs; use the original URL, size and SHA256.")
        if partial.is_file() and partial.stat().st_size == args.size:
            old_chunk_size = saved["chunk_size"]
            old_completed = set(saved.get("completed", []))
            if old_chunk_size == args.chunk_size:
                completed = old_completed
            else:
                # Keep already written byte intervals when switching to
                # smaller blocks. Only completely covered new blocks resume.
                for index in range((args.size + args.chunk_size - 1) // args.chunk_size):
                    begin = index * args.chunk_size
                    end = min(args.size, begin + args.chunk_size) - 1
                    covering = range(begin // old_chunk_size, end // old_chunk_size + 1)
                    if all(old_index in old_completed for old_index in covering):
                        completed.add(index)
                manifest_path.with_name(manifest_path.name + ".previous").write_text(json.dumps(saved, indent=2) + "\n")

    flags = os.O_CREAT | os.O_RDWR
    descriptor = os.open(partial, flags, 0o644)
    os.ftruncate(descriptor, args.size)
    lock = threading.Lock()
    redirect_lock = threading.Lock()
    transport_url = getattr(args, "transport_url", None) or args.url
    if getattr(args, "signed_url_file", None):
        transport_url = Path(args.signed_url_file).read_text().strip()
    resolved_url = transport_url
    stop = threading.Event()
    start_time = time.monotonic()
    bytes_this_run = 0
    chunks = (args.size + args.chunk_size - 1) // args.chunk_size

    def save_manifest():
        temporary = manifest_path.with_name(manifest_path.name + ".tmp")
        temporary.write_text(json.dumps({**metadata, "completed": sorted(completed)}, indent=2) + "\n")
        temporary.replace(manifest_path)

    def fetch(index):
        nonlocal bytes_this_run, resolved_url
        start = index * args.chunk_size
        end = min(args.size, start + args.chunk_size) - 1
        for attempt in range(args.retries):
            if stop.is_set() or STOP_REQUESTED.is_set():
                return False
            try:
                with redirect_lock:
                    current_url = resolved_url
                request = Request(unique_url(current_url, start, attempt), headers={
                    "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
                    "User-Agent": "VGGT-RK3588-Offline-Deployment/1.0",
                })
                with urlopen(request, timeout=args.timeout) as response:
                    expected_range = f"bytes {start}-{end}/{args.size}"
                    actual_range = response.headers.get("Content-Range", "")
                    if response.status != 206 or actual_range != expected_range:
                        raise InvalidRangeResponse(f"Invalid range response: HTTP {response.status}, {actual_range!r}; expected {expected_range!r}")
                    length = response.headers.get("Content-Length")
                    if length is not None and int(length) != end - start + 1:
                        raise RuntimeError("Range Content-Length differs from requested bytes.")
                    with redirect_lock:
                        # A signed redirect is reused only in this process.
                        # It remains protected by normal hostname TLS checks.
                        if urlsplit(response.url).hostname != urlsplit(current_url).hostname:
                            resolved_url = response.url
                    received = response.read(end - start + 2)
                    if len(received) != end - start + 1:
                        raise RuntimeError(f"Short/long range body: {len(received)} bytes")
                offset = 0
                while offset < len(received):
                    written = os.pwrite(descriptor, received[offset:], start + offset)
                    if written <= 0:
                        raise RuntimeError("Failed to write downloaded range.")
                    offset += written
                with lock:
                    os.fsync(descriptor)
                    completed.add(index)
                    bytes_this_run += len(received)
                    save_manifest()
                    elapsed = max(0.001, time.monotonic() - start_time)
                    progress = sum(min(args.chunk_size, args.size - chunk * args.chunk_size) for chunk in completed)
                    print(f"{len(completed)}/{chunks} ranges; {progress / 2**20:.1f}/{args.size / 2**20:.1f} MiB; {bytes_this_run / elapsed / 2**20:.2f} MiB/s", flush=True)
                return True
            except Exception as exc:
                if (isinstance(exc, HTTPError) and exc.code in {401, 403}) or isinstance(exc, InvalidRangeResponse):
                    with redirect_lock:
                        # Expired CDN signatures refresh through the original
                        # transport endpoint on the next attempt.
                        resolved_url = getattr(args, "transport_url", None) or args.url
                print(f"Range {index}, attempt {attempt + 1}/{args.retries}: {type(exc).__name__}: {exc}", flush=True)
                if attempt + 1 == args.retries:
                    # Leave this range pending while the rest of the queue
                    # advances. The next round retries only missing ranges.
                    return False
                # An Event wait lets another failed range interrupt retries.
                stop.wait(min(10, 1 + attempt * 2))

    pending = [index for index in range(chunks) if index not in completed]
    print(f"Downloading official VGGT: {destination}; {len(completed)}/{chunks} ranges resumed", flush=True)
    try:
        round_index = 0
        while pending and not STOP_REQUESTED.is_set():
            round_index += 1
            print(f"Round {round_index}: {len(pending)} ranges pending", flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(fetch, index) for index in pending]
                for future in as_completed(futures):
                    if STOP_REQUESTED.is_set():
                        for queued in futures:
                            queued.cancel()
                        break
                    future.result()
            pending = [index for index in range(chunks) if index not in completed]
            if pending:
                maximum_rounds = getattr(args, "rounds", 0)
                if maximum_rounds and round_index >= maximum_rounds:
                    break
                print(f"Round {round_index} finished; {len(pending)} missing ranges will retry", flush=True)
                STOP_REQUESTED.wait(2)
        if len(completed) != chunks:
            raise RuntimeError("Not all ranges finished; rerun this command to resume.")
    finally:
        stop.set()
        os.close(descriptor)

    print("All ranges complete; verifying official SHA256...", flush=True)
    actual_sha256 = sha256_file(partial)
    if actual_sha256 != args.sha256:
        raise RuntimeError(f"SHA256 mismatch: {actual_sha256}; expected {args.sha256}. Partial file preserved.")
    partial.replace(destination)
    print(f"VERIFIED size={destination.stat().st_size} SHA256={actual_sha256} file={destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="External SSD destination, e.g. /media/.../weights/model.pt")
    parser.add_argument("--url", default=MODEL_URL)
    parser.add_argument("--transport-url", help="Alternate download endpoint; verification still uses --size/--sha256")
    parser.add_argument("--signed-url-file", help="A saved official redirect URL; refreshed on signature expiry")
    parser.add_argument("--size", type=int, default=MODEL_SIZE)
    parser.add_argument("--sha256", default=MODEL_SHA256)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--timeout", type=float, default=40)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=0, help="Retry rounds; 0 continues until verified or interrupted")
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_size < 1 or args.size < 1 or not re.fullmatch(r"[0-9a-f]{64}", args.sha256):
        parser.error("workers, chunk-size and size must be positive; sha256 must have 64 lowercase hex digits")
    install_process_dns_cache()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _signum, _frame: STOP_REQUESTED.set())
    download(args)


if __name__ == "__main__":
    main()
