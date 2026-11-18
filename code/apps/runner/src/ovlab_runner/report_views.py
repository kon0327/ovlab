"""Short local report paths for browsers opening experiment exports over Windows/WSL."""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import uuid
from urllib.parse import quote, unquote, urlsplit

VIEW_VERSION = 2


def _source(root, candidate):
    root = Path(root).resolve()
    target = Path(candidate).resolve()
    if not target.is_relative_to(root):
        return None
    if target.is_file() and target.suffix == ".json":
        target = target.parent / "index.html"
    elif target.is_dir():
        target = target / "index.html"
    return target if target.is_file() and target.suffix == ".html" else None


def _target(root, page):
    key = hashlib.sha256(page.relative_to(root).as_posix().encode()).hexdigest()[:12]
    return root / "v" / key


def _fingerprint(page):
    identity = page.parent / ("manifest.json" if (page.parent / "manifest.json").is_file() else "metadata.json")
    return hashlib.sha256(page.read_bytes() + (identity.read_bytes() if identity.is_file() else b"")).hexdigest()


def report_page(root, candidate):
    """Resolve only real HTML pages, preferring a current short browser view."""
    if not candidate:
        return None
    root = Path(root).resolve()
    page = _source(root, candidate)
    if page is None:
        return None
    target = _target(root, page)
    if target.is_symlink() or not target.resolve().is_relative_to(root):
        return page
    try:
        document = json.loads((target / "view.json").read_text())
        inventory = document.get("files", [])
        intact = bool(inventory) and all(
            isinstance(item, dict) and isinstance(item.get("path"), str)
            and (target / item["path"]).resolve().is_relative_to(target.resolve())
            and (target / item["path"]).is_file()
            for item in inventory)
        if document.get("view_version") == VIEW_VERSION and document.get("source_sha256") == _fingerprint(page) and (target / "index.html").is_file() and intact:
            return target / "index.html"
    except (OSError, ValueError):
        pass
    return page


def _navigation(contents, directory, root):
    href = quote(os.path.relpath(root / 'index.html', directory), safe='/')
    anchor = f'<a class="experiment-return" href="{html.escape(href, quote=True)}">← Experiment overview</a>'
    if '<div class="related">' in contents:
        return contents.replace('<div class="related">', '<div class="related">' + anchor, 1)
    navigation = '<nav aria-label="Experiment navigation" style="padding:.5rem 1.25rem;margin:0;background:#eaf2fb;font:600 14px/1.5 system-ui,sans-serif"><span style="display:inline-flex;align-items:center;min-height:44px">' + anchor + '</span></nav>'
    return contents.replace('</header>', '</header>' + navigation, 1)


def publish_report_view(root, candidate):
    """Create a disposable, checksum-inventoried copy; never change source evidence."""
    root = Path(root).resolve()
    page = _source(root, candidate)
    if page is None:
        return None
    target = _target(root, page)
    if target.is_symlink():
        raise ValueError("report view target must not be a symbolic link")
    if report_page(root, page) == target / "index.html":
        return target / "index.html"
    files = sorted(x for x in page.parent.rglob('*') if x.is_file())
    if any(x.is_symlink() for x in page.parent.rglob('*')):
        raise ValueError("report view source contains a symbolic link")
    # Interactive dashboards have internal JS paths. Their relative paths are
    # already short; preserve them. Static export browsers flatten long filenames.
    interactive = (page.parent / "assets/dashboard.js").is_file()
    mapping = {x.relative_to(page.parent).as_posix():
               (x.relative_to(page.parent).as_posix() if interactive else
                'index.html' if x == page else 'a/' + hashlib.sha256(x.relative_to(page.parent).as_posix().encode()).hexdigest()[:16] + x.suffix)
               for x in files}
    stage = target.parent / f'.{target.name}.{uuid.uuid4().hex}.partial'
    backup = target.parent / f'.{target.name}.{uuid.uuid4().hex}.previous'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink():
        raise ValueError("report view directory must not be a symbolic link")
    stage.mkdir()
    try:
        for original in files:
            relative = original.relative_to(page.parent).as_posix()
            destination = stage / mapping[relative]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if original == page and not interactive:
                def rewrite(match):
                    url = html.unescape(match.group(3))
                    parsed = urlsplit(url)
                    path = unquote(parsed.path)
                    if parsed.scheme or parsed.netloc or not path:
                        return match.group(0)
                    resolved = (page.parent / path).resolve()
                    if not resolved.is_relative_to(page.parent):
                        return match.group(0)
                    key = resolved.relative_to(page.parent).as_posix()
                    if key not in mapping:
                        return match.group(0)
                    replacement = quote(mapping[key], safe='/')
                    if parsed.fragment:
                        replacement += '#' + parsed.fragment
                    return match.group(1) + match.group(2) + html.escape(replacement, quote=True) + match.group(2)
                contents = re.sub(r'((?:href|src)=)(["\x27])(.*?)\2', rewrite, original.read_text(encoding='utf-8'))
                destination.write_text(_navigation(contents, target / Path(mapping[relative]).parent, root), encoding='utf-8')
            elif original.suffix == '.html':
                contents = _navigation(original.read_text(encoding='utf-8'), target / Path(mapping[relative]).parent, root)
                destination.write_text(contents, encoding='utf-8')
            elif interactive and original.suffix == '.js' and b'"video_href"' in original.read_bytes():
                # Episode payload video paths were relative to the original
                # build. Give immutable canonical videos a short local copy too.
                def video(match):
                    value = json.loads(match.group(1))
                    canonical = (page.parent / value).resolve()
                    allowed = (root.parent.parent / 'runs' / root.name).resolve()
                    if not canonical.is_relative_to(allowed) or not canonical.is_file():
                        return match.group(0)
                    name = 'a/video-' + hashlib.sha256(str(canonical).encode()).hexdigest()[:16] + canonical.suffix
                    copy = stage / name
                    copy.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(canonical, copy)
                    return '"video_href":' + json.dumps(name)
                contents = re.sub(r'"video_href"\s*:\s*("(?:[^"\\]|\\.)*")', video, original.read_text(encoding='utf-8'))
                destination.write_text(contents, encoding='utf-8')
            else:
                shutil.copyfile(original, destination)
        inventory = [{"path": x.relative_to(stage).as_posix(), "sha256": hashlib.sha256(x.read_bytes()).hexdigest()}
                     for x in sorted(stage.rglob('*')) if x.is_file()]
        (stage / 'view.json').write_text(json.dumps({"view_version": VIEW_VERSION, "source": page.relative_to(root).as_posix(),
             "source_sha256": _fingerprint(page), "files": inventory}, sort_keys=True), encoding='utf-8')
        for item in stage.rglob('*'):
            item.chmod(0o2770 if item.is_dir() else 0o640)
        stage.chmod(0o2770)
        if target.exists():
            target.rename(backup)
        try:
            stage.rename(target)
        except BaseException:
            if backup.exists():
                backup.rename(target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return target / 'index.html'
