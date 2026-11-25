"""Reusable explicit dataset HTTP transfer; no provider discovery or storage orchestration."""
import os
from pathlib import Path
import time
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener
from .datasets import DatasetSecurityError, DatasetIntegrityError, DatasetInterruptedError, DatasetUnavailableError, DatasetProgress as Progress
from .artifact_identity import sha256_file, redact_url

class _HttpMixin:
    timeout_s = 30.0
    retries = 3

    @staticmethod
    def _validate_url(url: str, *, allow_local_http: bool = False) -> None:
        parsed = urlsplit(url)
        if parsed.username is not None or parsed.password is not None:
            raise DatasetSecurityError("dataset URL must not embed credentials")
        if parsed.scheme != "https":
            local = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
            if not (allow_local_http and local):
                raise DatasetSecurityError("dataset URL must use HTTPS")
        if not parsed.hostname:
            raise DatasetSecurityError("dataset URL requires a hostname")

    def _open(self, url: str, *, allow_local_http: bool = False):
        self._validate_url(url, allow_local_http=allow_local_http)
        response = build_opener().open(Request(url, headers={"User-Agent": "OVLAB-dataset/1"}), timeout=self.timeout_s)
        final = response.geturl()
        self._validate_url(final, allow_local_http=allow_local_http)
        return response

    def _download(
        self,
        url: str,
        target: Path,
        *,
        expected_sha256: str | None,
        allow_local_http: bool = False,
        progress: Progress | None = None,
        maximum_bytes: int = 256 * 1024 * 1024 * 1024,
    ) -> tuple[str, str, int]:
        partial = target.with_name(target.name + ".partial")
        target.parent.mkdir(parents=True, exist_ok=True)
        last_error: BaseException | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with self._open(url, allow_local_http=allow_local_http) as response, partial.open("wb") as stream:
                    final_url = response.geturl()
                    total = 0
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        total += len(block)
                        if total > maximum_bytes:
                            raise DatasetSecurityError("dataset download exceeds configured size limit")
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                actual = sha256_file(partial)
                if expected_sha256 is not None and actual != expected_sha256:
                    raise DatasetIntegrityError(
                        f"download checksum mismatch: expected {expected_sha256}, got {actual}"
                    )
                os.replace(partial, target)
                if progress is not None:
                    progress(f"downloaded {target.name} ({total} bytes)")
                return actual, redact_url(final_url), total
            except KeyboardInterrupt as exc:
                raise DatasetInterruptedError("dataset download interrupted; partial bytes remain staged") from exc
            except DatasetIntegrityError:
                partial.unlink(missing_ok=True)
                raise
            except (OSError, URLError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2 ** (attempt - 1), 4))
        raise DatasetUnavailableError(f"dataset download failed after {self.retries} attempts: {last_error}")
