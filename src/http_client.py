from __future__ import annotations

from contextlib import contextmanager
import os
import time
from typing import Any

import requests

from .config import DataConfig


class RequestClient:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Referer": "https://quote.eastmoney.com/",
    }

    def __init__(self, config: DataConfig):
        self.config = config
        self._last_request_at = 0.0

    def get_json(
        self,
        url: str,
        params: dict[str, Any],
        headers: dict[str, str] | None = None,
        trust_env: bool = False,
    ) -> Any:
        text = self.get_text(url, params=params, headers=headers, trust_env=trust_env)
        return requests.models.complexjson.loads(text)

    def get_text(
        self,
        url: str,
        params: dict[str, Any],
        headers: dict[str, str] | None = None,
        trust_env: bool = False,
    ) -> str:
        errors: list[str] = []
        for _ in range(self.config.retries + 1):
            self._throttle()
            try:
                with self._proxy_context(disable=(self.config.disable_proxy and not trust_env)):
                    with requests.Session() as session:
                        session.trust_env = trust_env
                        response = session.get(
                            url,
                            params=params,
                            headers=headers or self.headers,
                            timeout=self.config.timeout,
                        )
                        response.raise_for_status()
                        text = response.text.lstrip()
                        if text.startswith("<"):
                            raise RuntimeError("upstream returned HTML")
                        self._last_request_at = time.monotonic()
                        return response.text
            except Exception as exc:
                errors.append(str(exc))
                time.sleep(self.config.min_interval)
        raise RuntimeError("request failed: " + " | ".join(errors))

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.config.min_interval:
            time.sleep(self.config.min_interval - elapsed)

    @contextmanager
    def _proxy_context(self, disable: bool):
        if not disable:
            yield
            return

        keys = [
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "NO_PROXY",
            "no_proxy",
        ]
        snapshot = {key: os.environ.get(key) for key in keys}
        try:
            for key in keys:
                os.environ.pop(key, None)
            os.environ["NO_PROXY"] = "*"
            os.environ["no_proxy"] = "*"
            yield
        finally:
            for key in keys:
                value = snapshot.get(key)
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
