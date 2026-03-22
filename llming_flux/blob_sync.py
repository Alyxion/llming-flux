"""Flux Blob Sync — push/pull knowledge data to/from Azure Blob Storage.

Stores LISA and Linus data as individual blobs under a ``flux`` container:
    flux/lisa/lisa.db
    flux/lisa/image_hints.json
    flux/lisa/images/...
    flux/lisa/pdfs/...
    flux/linus/linus.db
    flux/linus/image_hints.json
    flux/linus/images/...
    flux/linus/pdfs/...

Upload (push):
    poetry run python -m llming_flux.blob_sync push --agent lisa
    poetry run python -m llming_flux.blob_sync push --agent linus
    poetry run python -m llming_flux.blob_sync push              # both

Download (pull) — called at startup for minimal container footprint:
    from llming_flux.blob_sync import ensure_data
    await ensure_data("lisa", data_dir)
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, quote

logger = logging.getLogger("flux.blob_sync")

# Files to sync (relative to data_dir)
_DATA_GLOBS = ["*.db", "*.json", "images/**/*", "pdfs/**/*"]
_MANIFEST = ".flux_manifest.json"


def _get_sas_parts(sas_url: str | None = None) -> tuple[str, str] | None:
    """Parse FLUX_BLOB_SAS into (base_url, sas_query).

    Returns (base_container_url, sas_token_query) or None.
    """
    sas = sas_url or os.environ.get("FLUX_BLOB_SAS", "").strip()
    if not sas:
        return None
    parsed = urlparse(sas)
    base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return base, parsed.query


def _blob_url(base: str, sas_query: str, blob_path: str) -> str:
    """Build full URL for a blob: base/blob_path?sas_query."""
    # quote the blob path segments but keep /
    quoted = "/".join(quote(seg, safe="") for seg in blob_path.split("/"))
    return f"{base}/{quoted}?{sas_query}"


def _file_md5(path: Path) -> str:
    """Compute MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(256 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Push (upload) ────────────────────────────────────────


def push(agent: str, data_dir: Path, sas_url: str | None = None, dry_run: bool = False) -> dict:
    """Upload data_dir contents to blob storage under flux/{agent}/.

    Returns {"uploaded": N, "skipped": N, "bytes": N}.
    """
    import requests

    parts = _get_sas_parts(sas_url)
    if not parts:
        raise RuntimeError("FLUX_BLOB_SAS not configured")
    base, sas_query = parts

    # Collect all files
    files: list[Path] = []
    for pattern in _DATA_GLOBS:
        files.extend(p for p in data_dir.glob(pattern) if p.is_file())

    # Load existing manifest (if any) to skip unchanged files
    manifest_url = _blob_url(base, sas_query, f"{agent}/{_MANIFEST}")
    old_manifest: dict[str, str] = {}
    try:
        resp = requests.get(manifest_url, timeout=30)
        if resp.status_code == 200:
            old_manifest = resp.json()
    except Exception:
        pass

    new_manifest: dict[str, str] = dict(old_manifest)  # start from old, update on success
    stats = {"uploaded": 0, "skipped": 0, "bytes": 0}

    for fp in sorted(files):
        rel = fp.relative_to(data_dir).as_posix()
        md5 = _file_md5(fp)

        if old_manifest.get(rel) == md5:
            new_manifest[rel] = md5
            stats["skipped"] += 1
            logger.debug("[PUSH] Unchanged: %s/%s", agent, rel)
            continue

        blob_path = f"{agent}/{rel}"
        url = _blob_url(base, sas_query, blob_path)

        if dry_run:
            logger.info("[PUSH] Would upload: %s (%d KB)", blob_path, fp.stat().st_size // 1024)
            stats["uploaded"] += 1
            stats["bytes"] += fp.stat().st_size
            continue

        # Determine content type
        ct = "application/octet-stream"
        lo = fp.name.lower()
        if lo.endswith(".json"):
            ct = "application/json"
        elif lo.endswith(".db"):
            ct = "application/x-sqlite3"
        elif lo.endswith(".pdf"):
            ct = "application/pdf"
        elif lo.endswith(".png"):
            ct = "image/png"
        elif lo.endswith(".jpg") or lo.endswith(".jpeg"):
            ct = "image/jpeg"
        elif lo.endswith(".svg"):
            ct = "image/svg+xml"

        data = fp.read_bytes()
        resp = requests.put(
            url,
            data=data,
            headers={
                "x-ms-blob-type": "BlockBlob",
                "Content-Type": ct,
            },
            timeout=300,
        )
        if resp.status_code in (200, 201):
            stats["uploaded"] += 1
            stats["bytes"] += len(data)
            new_manifest[rel] = md5  # only update manifest on success
            logger.info("[PUSH] Uploaded: %s (%d KB)", blob_path, len(data) // 1024)
        else:
            logger.error("[PUSH] Failed %s: HTTP %d %s", blob_path, resp.status_code, resp.text[:200])

    # Upload manifest
    if not dry_run and stats["uploaded"] > 0:
        manifest_data = json.dumps(new_manifest, indent=2).encode()
        resp = requests.put(
            manifest_url,
            data=manifest_data,
            headers={
                "x-ms-blob-type": "BlockBlob",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        if resp.status_code in (200, 201):
            logger.info("[PUSH] Manifest updated (%d entries)", len(new_manifest))
        else:
            logger.error("[PUSH] Manifest upload failed: %d", resp.status_code)
    elif not dry_run:
        # Nothing changed but ensure manifest exists
        if not old_manifest:
            manifest_data = json.dumps(new_manifest, indent=2).encode()
            requests.put(
                manifest_url,
                data=manifest_data,
                headers={"x-ms-blob-type": "BlockBlob", "Content-Type": "application/json"},
                timeout=30,
            )

    return stats


# ── Pull (download) ──────────────────────────────────────


async def ensure_data(agent: str, data_dir: Path, sas_url: str | None = None) -> bool:
    """Ensure data_dir has all files for the given agent.

    Downloads missing/changed files from blob storage.
    Returns True if data is ready, False if blob is unavailable.
    """
    import aiohttp

    parts = _get_sas_parts(sas_url)
    if not parts:
        # No blob configured — assume local data exists
        return data_dir.exists() and any(data_dir.glob("*.db"))
    base, sas_query = parts

    # Fetch remote manifest
    manifest_url = _blob_url(base, sas_query, f"{agent}/{_MANIFEST}")
    remote_manifest: dict[str, str] = {}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(manifest_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    remote_manifest = await resp.json()
                else:
                    logger.warning("[PULL] No manifest for %s (HTTP %d), skipping", agent, resp.status)
                    return data_dir.exists() and any(data_dir.glob("*.db"))
    except Exception as e:
        logger.warning("[PULL] Cannot reach blob for %s: %s", agent, e)
        return data_dir.exists() and any(data_dir.glob("*.db"))

    if not remote_manifest:
        return data_dir.exists() and any(data_dir.glob("*.db"))

    # Build local manifest for comparison
    data_dir.mkdir(parents=True, exist_ok=True)
    to_download: list[str] = []
    for rel, md5 in remote_manifest.items():
        local = data_dir / rel
        if local.exists() and _file_md5(local) == md5:
            continue
        to_download.append(rel)

    if not to_download:
        logger.info("[PULL] %s data up to date (%d files)", agent, len(remote_manifest))
        return True

    logger.info("[PULL] %s: downloading %d/%d files", agent, len(to_download), len(remote_manifest))

    async with aiohttp.ClientSession() as session:
        for rel in to_download:
            blob_path = f"{agent}/{rel}"
            url = _blob_url(base, sas_query, blob_path)
            local = data_dir / rel
            local.parent.mkdir(parents=True, exist_ok=True)

            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                    if resp.status != 200:
                        logger.error("[PULL] Failed %s: HTTP %d", blob_path, resp.status)
                        continue
                    tmp = local.with_suffix(local.suffix + ".tmp")
                    with open(tmp, "wb") as f:
                        async for chunk in resp.content.iter_chunked(256 * 1024):
                            f.write(chunk)
                    tmp.rename(local)
                    logger.info("[PULL] Downloaded: %s (%d KB)", rel, local.stat().st_size // 1024)
            except Exception as e:
                logger.error("[PULL] Error downloading %s: %s", blob_path, e)

    logger.info("[PULL] %s sync complete", agent)
    return any(data_dir.glob("*.db"))


# ── CLI ──────────────────────────────────────────────────


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Flux knowledge blob sync")
    parser.add_argument("action", choices=["push", "pull"], help="Push local→blob or pull blob→local")
    parser.add_argument("--agent", choices=["lisa", "linus"], help="Specific agent (default: both)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded (push only)")
    args = parser.parse_args()

    # Load .env
    from dotenv import load_dotenv
    load_dotenv()

    agents = [args.agent] if args.agent else ["lisa", "linus"]

    # Resolve data directories
    flux_root = Path(__file__).parent.parent.parent  # → dependencies/
    agent_dirs = {
        "lisa": flux_root / "AgenticLISA" / "data",
        "linus": flux_root / "Linus" / "data",
    }

    if args.action == "push":
        for agent in agents:
            data_dir = agent_dirs[agent]
            if not data_dir.exists():
                logger.error("Data dir not found: %s", data_dir)
                continue
            logger.info("=== Pushing %s from %s ===", agent, data_dir)
            stats = push(agent, data_dir, dry_run=args.dry_run)
            logger.info("  Uploaded: %d, Skipped: %d, Bytes: %d KB",
                        stats["uploaded"], stats["skipped"], stats["bytes"] // 1024)

    elif args.action == "pull":
        import asyncio
        for agent in agents:
            data_dir = agent_dirs[agent]
            logger.info("=== Pulling %s to %s ===", agent, data_dir)
            ok = asyncio.run(ensure_data(agent, data_dir))
            logger.info("  Ready: %s", ok)


if __name__ == "__main__":
    main()
