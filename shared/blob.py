"""Azure Blob helpers: upload, download, mirror a directory.

Auth: **AAD only** (this storage account forbids key-based / connection-string auth —
`KeyBasedAuthenticationNotPermitted`). We use `DefaultAzureCredential`, which resolves to:
    - locally: Azure CLI login (`az login`)
    - on the H100 VM: the VM's system-assigned **managed identity**
      (granted "Storage Blob Data Contributor" by `h100.up()`).
No connection string is ever used.

All functions take blob *names* that already include the desired prefix; use the
prefixes in `config.BLOB_LAYOUT` to keep storage organized.

CLI:
    python -m nymt_shared.blob upload <local> <blob_name>
    python -m nymt_shared.blob download <blob_name> <local>
    python -m nymt_shared.blob mirror <local_dir> <blob_prefix>
    python -m nymt_shared.blob ls [prefix]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from . import config


def _credential():
    """AAD credential usable both locally (az login) and on the VM (managed identity)."""
    from azure.identity import DefaultAzureCredential

    return DefaultAzureCredential(exclude_interactive_browser_credential=True)


def _service_client():
    from azure.storage.blob import BlobServiceClient

    account_url = f"https://{config.STORAGE_ACCOUNT}.blob.core.windows.net"
    return BlobServiceClient(account_url, credential=_credential())


def _container_client(container: str | None = None):
    svc = _service_client()
    return svc.get_container_client(container or config.CONTAINER)


def upload_file(local_path: str | Path, blob_name: str, container: str | None = None,
                overwrite: bool = True) -> str:
    """Upload one file. Returns the blob name."""
    cc = _container_client(container)
    local_path = Path(local_path)
    with open(local_path, "rb") as fh:
        cc.upload_blob(name=blob_name, data=fh, overwrite=overwrite)
    return blob_name


def download_file(blob_name: str, local_path: str | Path, container: str | None = None) -> Path:
    """Download one blob to a local path (parents created)."""
    cc = _container_client(container)
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as fh:
        stream = cc.download_blob(blob_name)
        fh.write(stream.readall())
    return local_path


def list_blobs(prefix: str = "", container: str | None = None) -> list[str]:
    cc = _container_client(container)
    return [b.name for b in cc.list_blobs(name_starts_with=prefix)]


def blob_exists(blob_name: str, container: str | None = None) -> bool:
    cc = _container_client(container)
    return cc.get_blob_client(blob_name).exists()


def _iter_files(local_dir: Path, exclude: Iterable[str]) -> Iterable[Path]:
    exclude = set(exclude)
    for p in local_dir.rglob("*"):
        if not p.is_file():
            continue
        if any(part in exclude for part in p.parts):
            continue
        yield p


def mirror_dir(local_dir: str | Path, blob_prefix: str, container: str | None = None,
               exclude: Iterable[str] = (".git", "__pycache__", ".venv"),
               only_new: bool = True, verbose: bool = True) -> int:
    """Upload everything under `local_dir` to `blob_prefix/<relpath>`.

    If `only_new` is True, skips blobs that already exist with the same size
    (cheap idempotent mirror for the background snapshot loop).
    Returns the number of files uploaded.
    """
    local_dir = Path(local_dir)
    if not local_dir.exists():
        return 0
    cc = _container_client(container)
    prefix = blob_prefix.rstrip("/") + "/"

    existing: dict[str, int] = {}
    if only_new:
        for b in cc.list_blobs(name_starts_with=prefix):
            existing[b.name] = b.size

    n = 0
    for f in _iter_files(local_dir, exclude):
        rel = f.relative_to(local_dir).as_posix()
        name = prefix + rel
        size = f.stat().st_size
        if only_new and existing.get(name) == size:
            continue
        with open(f, "rb") as fh:
            cc.upload_blob(name=name, data=fh, overwrite=True)
        n += 1
        if verbose:
            print(f"[mirror] {name} ({size} B)")
    return n


def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="nymt_shared.blob")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("upload"); p.add_argument("local"); p.add_argument("blob_name")
    p = sub.add_parser("download"); p.add_argument("blob_name"); p.add_argument("local")
    p = sub.add_parser("mirror"); p.add_argument("local_dir"); p.add_argument("blob_prefix")
    p.add_argument("--all", action="store_true", help="re-upload even unchanged files")
    p = sub.add_parser("ls"); p.add_argument("prefix", nargs="?", default="")
    args = ap.parse_args(argv)

    if args.cmd == "upload":
        print(upload_file(args.local, args.blob_name))
    elif args.cmd == "download":
        print(download_file(args.blob_name, args.local))
    elif args.cmd == "mirror":
        n = mirror_dir(args.local_dir, args.blob_prefix, only_new=not args.all)
        print(f"uploaded {n} file(s)")
    elif args.cmd == "ls":
        for name in list_blobs(args.prefix):
            print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
