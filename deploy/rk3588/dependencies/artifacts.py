#!/usr/bin/env python3
"""Download and verify the pinned RK3588S offline environment."""

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import zipfile


HERE = Path(__file__).resolve().parent
CHUNK_SIZE = 2 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid(path, spec):
    return (
        path.is_file()
        and path.stat().st_size == spec["size"]
        and sha256(path) == spec["sha256"]
    )


def artifact_path(root, spec):
    relative = Path(spec["directory"]) / spec["filename"]
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Artifact paths must remain under --output")
    return root / relative


def download_chunk(spec, root, start, end, retries, resolves):
    chunks = root / "tmp" / "dependency-downloads" / spec["sha256"]
    chunks.mkdir(parents=True, exist_ok=True)
    path = chunks / f"{start}-{end}.chunk"
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return path
    for attempt in range(retries):
        command = [
            "curl", "-fsSL", "--proto", "=https", "--proto-redir", "=https",
            "--connect-timeout", "15", "--max-time", "60",
            "--speed-limit", "1000", "--speed-time", "15",
            "--range", f"{start}-{end}", "--output", str(path),
        ]
        for address in resolves:
            command.extend(["--resolve", address])
        command.append(spec["url"])
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0 and path.exists() and path.stat().st_size == expected:
            return path
        detail = result.stderr.strip()[-500:] or "Unexpected response length"
        print(f"retry {spec['filename']} {start}-{end} ({attempt + 1}/{retries}): {detail}", flush=True)
        if attempt + 1 < retries:
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Download failed; retained chunks for {spec['filename']}")


def fetch(spec, root, args):
    final = artifact_path(root, spec)
    if valid(final, spec):
        print(f"verified existing {spec['filename']}", flush=True)
        return final
    if "url" not in spec:
        raise ValueError(f"No download URL for {spec['filename']}")
    final.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {spec['filename']} ({spec['size']} bytes)", flush=True)
    ranges = [(start, min(start + CHUNK_SIZE - 1, spec["size"] - 1))
              for start in range(0, spec["size"], CHUNK_SIZE)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(download_chunk, spec, root, start, end, args.retries, args.resolve)
                   for start, end in ranges]
        paths = [future.result() for future in futures]
    assembled = final.with_name(final.name + ".part")
    with assembled.open("wb") as output:
        for path in paths:
            with path.open("rb") as part:
                shutil.copyfileobj(part, output, length=1024 * 1024)
    if not valid(assembled, spec):
        raise RuntimeError(f"SHA256 mismatch: {spec['filename']}; retained chunks for inspection")
    assembled.replace(final)
    print(f"ready {spec['filename']}", flush=True)
    return final


def build_antlr(spec, source_path, root):
    """Package verified upstream files without running setup.py or installing tools."""
    if not valid(source_path, spec["build_from"]):
        raise RuntimeError("ANTLR source archive failed SHA256 verification")
    template = json.loads((HERE / spec["build_template"]).read_text())
    final = artifact_path(root, spec)
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.with_name(final.name + ".part")
    with tarfile.open(source_path, "r:gz") as source, zipfile.ZipFile(temporary, "w") as wheel:
        wheel.comment = base64.b64decode(template["archive_comment_base64"])
        for item in template["members"]:
            if "source_member" in item:
                member = source.getmember(item["source_member"])
                if not member.isfile() or member.size != item["size"]:
                    raise RuntimeError(f"Unexpected ANTLR source member: {member.name}")
                with source.extractfile(member) as stream:
                    data = stream.read()
            else:
                data = item["text"].encode("utf-8")
            if len(data) != item["size"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise RuntimeError(f"ANTLR wheel member mismatch: {item['filename']}")
            info = zipfile.ZipInfo(item["filename"], tuple(item["zip"]["date_time"]))
            for key, value in item["zip"].items():
                if key == "date_time":
                    continue
                if key.endswith("_base64"):
                    setattr(info, key.removesuffix("_base64"), base64.b64decode(value))
                else:
                    setattr(info, key, value)
            wheel.writestr(info, data)
    if not valid(temporary, spec):
        raise RuntimeError("Rebuilt ANTLR archive differs from the pinned wheel SHA256")
    temporary.replace(final)
    print(f"rebuilt and verified {spec['filename']}", flush=True)


def selected(manifest, kind):
    accepted = {"all": {"wheel", "python", "deb"}, "wheels": {"wheel"},
                "python": {"python"}, "debs": {"deb"}}[kind]
    return [item for item in manifest["artifacts"] if item["kind"] in accepted]


def write_install_files(root, manifest):
    wheelhouse = root / "wheelhouse"
    wheelhouse.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "requirements-board.txt", wheelhouse / "requirements-board.txt")
    for kind, directory in [("wheel", "wheelhouse"), ("deb", "debs")]:
        entries = [item for item in manifest["artifacts"] if item["kind"] == kind
                   and valid(artifact_path(root, item), item)]
        if entries:
            path = root / directory / "SHA256SUMS"
            path.write_text("".join(f"{item['sha256']}  {item['filename']}\n" for item in entries))


def download(args, manifest, root):
    for spec in selected(manifest, args.kind):
        if "build_from" in spec and not valid(artifact_path(root, spec), spec):
            source = fetch(spec["build_from"], root, args)
            build_antlr(spec, source, root)
        else:
            fetch(spec, root, args)
    write_install_files(root, manifest)


def check_lock(manifest):
    expected = {f"{spec['name']}=={spec['version']}" for spec in manifest["artifacts"]
                if spec["kind"] == "wheel"}
    actual = {line.strip() for line in (HERE / "requirements-board.txt").read_text().splitlines()
              if line.strip() and not line.lstrip().startswith("#")}
    if actual != expected:
        raise RuntimeError("requirements-board.txt differs from the artifact manifest")


def verify(args, manifest, root):
    check_lock(manifest)
    count = total = 0
    for spec in selected(manifest, args.kind):
        path = artifact_path(root, spec)
        if not valid(path, spec):
            raise RuntimeError(f"Missing artifact or size/SHA256 mismatch: {path}")
        if spec["kind"] == "wheel":
            with zipfile.ZipFile(path) as archive:
                bad = archive.testzip()
                if bad:
                    raise RuntimeError(f"Wheel ZIP CRC mismatch: {path}: {bad}")
        elif spec["kind"] == "python":
            with tarfile.open(path, "r:gz") as archive:
                for member in archive:
                    if member.isfile():
                        with archive.extractfile(member) as stream:
                            for _ in iter(lambda: stream.read(1024 * 1024), b""):
                                pass
        elif spec["kind"] == "deb":
            with path.open("rb") as stream:
                if stream.read(8) != b"!<arch>\n":
                    raise RuntimeError(f"Invalid Debian archive: {path}")
        count += 1
        total += spec["size"]
    print(f"Verified {count} artifacts, {total} bytes; SHA256 and archive checks passed.", flush=True)
    if args.resolve_dependencies:
        if args.kind not in {"all", "wheels"}:
            raise ValueError("--resolve-dependencies requires --kind all or wheels")
        resolve_offline(root, manifest)


def resolve_offline(root, manifest):
    """Cross-platform pip dry run with ARM markers and all network calls blocked."""
    temporary = root / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temporary)
    os.environ["PIP_CACHE_DIR"] = str(root / "cache")
    import tempfile
    tempfile.tempdir = str(temporary)
    try:
        import pip._vendor.packaging.markers as markers
        from pip._internal.cli.main import main as pip_main
    except ImportError as error:
        raise RuntimeError("The host Python needs pip for --resolve-dependencies") from error

    original = markers.default_environment

    def target_environment():
        environment = original()
        environment.update(
            implementation_name="cpython", implementation_version="3.11.17",
            os_name="posix", platform_machine="aarch64",
            platform_python_implementation="CPython", platform_system="Linux",
            python_full_version="3.11.17", python_version="3.11", sys_platform="linux",
        )
        return environment

    markers.default_environment = target_environment

    def no_network(*unused_args, **unused_kwargs):
        raise RuntimeError("Network access is forbidden during offline verification")

    socket.getaddrinfo = no_network
    socket.create_connection = no_network
    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network
    report = root / "offline-install-plan.json"
    command = [
        "--isolated", "install", "--dry-run", "--ignore-installed", "--no-index",
        "--no-cache-dir", "--disable-pip-version-check", "--only-binary=:all:",
        "--find-links", str(root / "wheelhouse"), "--target", str(temporary / "offline-probe"),
        "--python-version", "311", "--implementation", "cp", "--abi", "cp311",
        "--report", str(report), "-r", str(HERE / "requirements-board.txt"),
    ]
    for platform in ["manylinux2014_aarch64", "manylinux_2_17_aarch64", "manylinux_2_27_aarch64",
                     "manylinux_2_28_aarch64", "manylinux_2_35_aarch64"]:
        command.extend(["--platform", platform])
    result = pip_main(command)
    if result:
        raise RuntimeError(f"Offline pip dependency resolution failed with exit code {result}")
    plan = json.loads(report.read_text())["install"]
    if len(plan) != sum(item["kind"] == "wheel" for item in manifest["artifacts"]):
        raise RuntimeError("Offline install plan has an unexpected package count")
    if any(not item["download_info"]["url"].startswith("file:") for item in plan):
        raise RuntimeError("Offline install plan contains a nonlocal artifact")
    print(f"Offline ARM64 / Python 3.11 pip resolution passed: {len(plan)} local wheels.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["download", "verify"])
    parser.add_argument("--output", type=Path, required=True, help="External artifact root containing wheelhouse/ and debs/")
    parser.add_argument("--kind", choices=["all", "wheels", "python", "debs"], default="all")
    parser.add_argument("--jobs", type=int, default=4, help="Concurrent 2 MiB range downloads per artifact")
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--resolve", action="append", default=[], help="Optional curl HOST:443:IP; TLS still verifies the official hostname")
    parser.add_argument("--resolve-dependencies", action="store_true", help="For verify: run a strictly offline pip dry run; no packages are installed")
    args = parser.parse_args()
    if args.jobs < 1 or args.retries < 1:
        parser.error("--jobs and --retries must be positive")
    if args.operation == "download" and args.resolve_dependencies:
        parser.error("--resolve-dependencies is a verify option")
    manifest = json.loads((HERE / "artifacts.json").read_text())
    root = args.output.expanduser().resolve()
    if args.operation == "download":
        root.mkdir(parents=True, exist_ok=True)
        check_lock(manifest)
        download(args, manifest, root)
    else:
        verify(args, manifest, root)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
