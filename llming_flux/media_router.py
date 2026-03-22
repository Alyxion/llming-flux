"""Generic FastAPI router for serving images and PDFs from a data directory."""

from pathlib import Path


def guess_mime(filename: str) -> str:
    lo = filename.lower()
    if lo.endswith(".png"): return "image/png"
    if lo.endswith(".gif"): return "image/gif"
    if lo.endswith(".webp"): return "image/webp"
    if lo.endswith(".svg"): return "image/svg+xml"
    if lo.endswith(".pdf"): return "application/pdf"
    return "image/jpeg"


def build_media_router(prefix: str, data_dir: Path, require_session: bool = True):
    """Build a FastAPI APIRouter serving images and PDFs.

    Args:
        prefix: URL prefix, e.g. ``"lisa-media"`` → ``/api/lisa-media/image/...``
        data_dir: Root data directory containing ``images/`` and ``pdfs/`` subdirs.
        require_session: If True, require ``request.session["id"]`` for auth.

    Returns:
        A ``fastapi.APIRouter`` instance.
    """
    from fastapi import APIRouter
    from fastapi.responses import FileResponse, Response
    from starlette.requests import Request

    router = APIRouter()
    _image_cache: dict[str, tuple[bytes, str]] = {}

    @router.get(f"/api/{prefix}/image/{{path:path}}")
    async def serve_image(path: str, request: Request):
        if require_session and not request.session.get("id"):
            return Response(status_code=401, content="Not authenticated")
        if ".." in path or path.startswith("/"):
            return Response(status_code=400, content="Invalid path")
        if path in _image_cache:
            data, ct = _image_cache[path]
            return Response(content=data, media_type=ct,
                            headers={"Cache-Control": "public, max-age=86400"})
        fp = data_dir / "images" / path
        if not fp.exists() or not fp.is_file():
            return Response(status_code=404, content="Image not found")
        ct = guess_mime(fp.name)
        data = fp.read_bytes()
        if len(data) < 2 * 1024 * 1024:
            _image_cache[path] = (data, ct)
        return Response(content=data, media_type=ct,
                        headers={"Cache-Control": "public, max-age=86400"})

    @router.get(f"/api/{prefix}/pdf/{{filename}}")
    async def serve_pdf(filename: str, request: Request):
        if require_session and not request.session.get("id"):
            return Response(status_code=401, content="Not authenticated")
        if ".." in filename or "/" in filename:
            return Response(status_code=400, content="Invalid filename")
        fp = data_dir / "pdfs" / filename
        if not fp.exists():
            return Response(status_code=404, content="PDF not found")
        return FileResponse(fp, media_type="application/pdf",
                            headers={"Cache-Control": "public, max-age=86400"})

    return router
