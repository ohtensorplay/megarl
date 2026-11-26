"""Network tool layer: safe HTTP fetch, pluggable web search and a tool proxy.

This complements :mod:`megarl.tools` with the agent-facing network surface:

* :func:`make_fetch_tool` — SSRF-hardened HTTP(S) GET with pinned DNS, bounded
  redirects, a hard response-size cap and content-type filtering.
* :func:`make_search_tool` — a search tool backed by a caller-supplied backend
  (a callback or a generic JSON search API). No fake results when unconfigured.
* :class:`ToolHTTPProxy` — stdlib HTTP server that exposes a :class:`ToolRegistry`
  to external agents (schema discovery, single/batch calls) and lets them
  register *HTTP-backed* handlers dynamically — never arbitrary code.

Everything uses the standard library so the wheel gains no new runtime
dependency. All network paths re-validate URLs and pin connections to an IP
that was checked at resolution time, which closes the DNS-rebinding window a
naive ``urlopen`` implementation leaves open.
"""

from __future__ import annotations

import dataclasses
import hmac
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol
from urllib.parse import urlsplit

from . import megarl_rs as _native
from .observability import Telemetry
from .tools import ParsedToolCall, Tool, ToolError, ToolRegistry


class NetworkToolError(ToolError):
    """A network tool failed (invalid URL, blocked host, timeout, bad payload)."""


@dataclasses.dataclass(frozen=True, slots=True)
class HTTPResponse:
    status: int
    url: str
    content_type: str
    text: str
    bytes_read: int
    truncated: bool


# ---------------------------------------------------------------------------
# Safe HTTP core — thin adapter over the Rust spine (MegaLM remote_media port)
# ---------------------------------------------------------------------------

def http_request(
    method: str,
    url: str,
    *,
    body: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout_secs: float = 10.0,
    max_bytes: int = 1 << 16,
    max_redirects: int = 3,
    allow_private: bool = False,
    allow_http: bool = True,
    allowed_domains: Sequence[str] = (),
) -> HTTPResponse:
    """Perform a bounded, SSRF-hardened HTTP request via Rust.

    DNS validation, IP pinning, redirect re-validation, retry classification,
    streaming size caps and structured errors all live in ``megarl-nettools``;
    this function only marshals a JSON config and decodes the result. The GIL
    is released by Rust while the request is in flight.
    """
    config = {
        "method": method.upper(),
        "url": url,
        "body_json": json.dumps(body, separators=(",", ":")) if body is not None else None,
        "headers": dict(headers or {}),
        "max_bytes": int(max_bytes),
        "max_redirects": int(max_redirects),
        "max_attempts": 2,
        "connect_timeout_secs": max(1, min(int(timeout_secs), 5)),
        "read_timeout_secs": max(1, int(timeout_secs)),
        "total_timeout_secs": max(1, int(timeout_secs) + 2),
        "allow_private": bool(allow_private),
        "allow_http": bool(allow_http),
        "allowed_domains": list(allowed_domains),
    }
    payload = json.loads(_native.http_request_json(json.dumps(config, separators=(",", ":"))))
    if not payload.get("ok"):
        error = payload["error"]
        raise NetworkToolError(f"{error['code']}: {error['message']}")
    response = payload["response"]
    return HTTPResponse(
        status=int(response["status"]),
        url=response["url"],
        content_type=response["content_type"],
        text=response["text"],
        bytes_read=int(response["bytes_read"]),
        truncated=bool(response["truncated"]),
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def make_fetch_tool(
    *,
    allow_private: bool = False,
    name: str = "fetch",
    timeout_secs: float = 10.0,
    max_bytes: int = 1 << 16,
) -> Tool:
    """Build an SSRF-hardened HTTP GET tool.

    ``allow_private=True`` is for localhost integration tests and on-premise
    deployments only; it deliberately relaxes the public-address check.
    """

    def fetch(arguments: dict[str, Any]) -> str:
        url = arguments["url"]
        limit = min(int(arguments.get("max_bytes", max_bytes)), max_bytes)
        timeout = min(int(arguments.get("timeout_ms", int(timeout_secs * 1000))) / 1000, timeout_secs)
        response = http_request(
            "GET", url, timeout_secs=timeout, max_bytes=limit, allow_private=allow_private
        )
        header = f"status={response.status} url={response.url} content_type={response.content_type}"
        if response.truncated:
            header += " [truncated]"
        return f"{header}\n{response.text}"

    return Tool(
        name=name,
        description="Fetch a public HTTP(S) URL and return its text body. Blocks private hosts.",
        parameters={
            "type": "object",
            "required": ["url"],
            "properties": {
                "url": {"type": "string", "minLength": 1, "maxLength": 2048},
                "max_bytes": {"type": "integer", "minimum": 1, "maximum": 262144},
                "timeout_ms": {"type": "integer", "minimum": 100, "maximum": 30000},
            },
            "additionalProperties": False,
        },
        handler=fetch,
        timeout_secs=timeout_secs + 1.0,
        max_result_chars=max_bytes + 512,
    )


class SearchBackend(Protocol):
    def search(self, query: str, limit: int) -> list[dict[str, str]]: ...


@dataclasses.dataclass(frozen=True, slots=True)
class CallbackSearchBackend:
    """Adapter for any callable ``(query, limit) -> list[dict]``."""

    callback: Callable[[str, int], Sequence[Mapping[str, str]]]

    def search(self, query: str, limit: int) -> list[dict[str, str]]:
        return [
            {str(key): str(value) for key, value in item.items()}
            for item in self.callback(query, limit)
        ]


@dataclasses.dataclass(frozen=True, slots=True)
class HTTPSearchBackend:
    """Generic JSON search API backend (POST) using the hardened HTTP core."""

    endpoint: str
    api_key_env: str | None = None
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer "
    extra_body: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    results_path: str = "results"
    title_path: str = "title"
    url_path: str = "url"
    content_path: str = "content"
    allow_private: bool = False
    timeout_secs: float = 10.0

    @staticmethod
    def _dig(value: Any, path: str) -> Any:
        for key in path.split("."):
            if not isinstance(value, Mapping) or key not in value:
                return None
            value = value[key]
        return value

    def search(self, query: str, limit: int) -> list[dict[str, str]]:
        import os

        body: dict[str, Any] = {"query": query, "max_results": limit, **self.extra_body}
        headers: dict[str, str] = {}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise NetworkToolError(f"search API key missing: {self.api_key_env}")
            headers[self.api_key_header] = f"{self.api_key_prefix}{key}"
        response = http_request(
            "POST",
            self.endpoint,
            body=body,
            headers=headers,
            timeout_secs=self.timeout_secs,
            max_bytes=1 << 18,
            allow_private=self.allow_private,
        )
        try:
            payload = json.loads(response.text)
        except ValueError as exc:
            raise NetworkToolError("search API returned invalid JSON") from exc
        results = self._dig(payload, self.results_path)
        if not isinstance(results, list):
            raise NetworkToolError(f"search API has no list at {self.results_path!r}")
        rendered: list[dict[str, str]] = []
        for item in results[:limit]:
            rendered.append(
                {
                    "title": str(self._dig(item, self.title_path) or ""),
                    "url": str(self._dig(item, self.url_path) or ""),
                    "content": str(self._dig(item, self.content_path) or ""),
                }
            )
        return rendered


def make_search_tool(backend: SearchBackend | None, *, name: str = "search") -> Tool:
    """Build a search tool; fails loudly when no backend is configured."""

    def search(arguments: dict[str, Any]) -> str:
        if backend is None:
            raise NetworkToolError("no search backend configured")
        limit = int(arguments.get("limit", 5))
        results = backend.search(str(arguments["query"]), limit)
        if not results:
            return "no results"
        lines = []
        for index, item in enumerate(results, 1):
            lines.append(
                f"[{index}] {item.get('title', '')}\n{item.get('url', '')}\n{item.get('content', '')}"
            )
        return "\n\n".join(lines)

    return Tool(
        name=name,
        description="Search the web for a query and return ranked titles, URLs and snippets.",
        parameters={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 512},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "additionalProperties": False,
        },
        handler=search,
        timeout_secs=15.0,
        max_result_chars=8192,
    )


def default_network_registry(
    *,
    search_backend: SearchBackend | None = None,
    allow_private: bool = False,
    telemetry: Telemetry | None = None,
) -> ToolRegistry:
    """Registry with the network tools; search is added only when configured."""
    tools: list[Tool] = [make_fetch_tool(allow_private=allow_private)]
    if search_backend is not None:
        tools.append(make_search_tool(search_backend))
    return ToolRegistry(tools, telemetry=telemetry)


# ---------------------------------------------------------------------------
# Agent↔environment HTTP proxy with dynamic (HTTP-backed) handlers
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class _RemoteSpec:
    name: str
    endpoint: str
    method: str
    headers: dict[str, str]
    allow_private: bool


class ToolHTTPProxy:
    """Expose a tool registry over HTTP with authenticated, dynamic handlers.

    Dynamic registration creates tools whose implementation is an HTTP call to
    a caller-supplied endpoint — never executable code — so an external agent
    can add environment capabilities without granting the proxy code execution.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        token: str | None = None,
        allow_dynamic: bool = True,
        allow_private_endpoints: bool = False,
        max_body_bytes: int = 1 << 18,
        max_in_flight: int = 8,
        telemetry: Telemetry | None = None,
    ) -> None:
        self.registry = registry
        self.host = host
        self.port = port
        self.token = token
        self.allow_dynamic = allow_dynamic
        self.allow_private_endpoints = allow_private_endpoints
        self.max_body_bytes = max_body_bytes
        self.telemetry = telemetry
        self._semaphore = threading.BoundedSemaphore(max(1, max_in_flight))
        self._specs: dict[str, _RemoteSpec] = {}
        self._spec_lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                if proxy.telemetry is not None:
                    proxy.telemetry.emit("tool_proxy_http", message=format % args)

            def _authorized(self) -> bool:
                if not proxy.token:
                    return True
                header = self.headers.get("Authorization", "")
                expected = f"Bearer {proxy.token}"
                return hmac.compare_digest(header, expected)

            def _send(self, status: int, payload: Any) -> None:
                body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _body(self) -> Any:
                length = self.headers.get("Content-Length")
                try:
                    size = int(length or 0)
                except ValueError as exc:
                    raise ValueError("invalid Content-Length") from exc
                if size < 0 or size > proxy.max_body_bytes:
                    raise ValueError("invalid or oversized request body")
                raw = self.rfile.read(size)
                return json.loads(raw or b"{}")

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                if self.path.split("?", 1)[0] == "/health":
                    proxy._send_health(self)  # intentionally unbounded / cheap
                    return
                if not self._authorized():
                    self._send(401, {"error": "unauthorized"})
                    return
                with proxy._semaphore:
                    if self.path.split("?", 1)[0] == "/tools":
                        self._send(200, {"tools": proxy.registry.schemas()})
                    else:
                        self._send(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                if not self._authorized():
                    self._send(401, {"error": "unauthorized"})
                    return
                path = self.path.split("?", 1)[0]
                try:
                    payload = self._body()
                except (ValueError, json.JSONDecodeError) as exc:
                    self._send(400, {"error": str(exc)})
                    return
                with proxy._semaphore:
                    if path == "/tools":
                        proxy._register(self, payload)
                    elif path == "/tools/call":
                        proxy._call_batch(self, payload)
                    elif path.startswith("/tools/") and path.endswith("/call"):
                        proxy._call_one(self, path[len("/tools/") : -len("/call")], payload)
                    else:
                        self._send(404, {"error": "not found"})

            def do_DELETE(self) -> None:  # noqa: N802
                if not self._authorized():
                    self._send(401, {"error": "unauthorized"})
                    return
                prefix, _, name = self.path.partition("/tools/")
                if prefix or not name:
                    self._send(404, {"error": "not found"})
                    return
                with proxy._semaphore, proxy._spec_lock:
                    spec = proxy._specs.pop(name, None)
                    if spec is not None:
                        proxy.registry.unregister(name)
                if spec is None:
                    self._send(404, {"error": "unknown dynamic tool"})
                    return
                self._send(200, {"ok": True, "name": name})

        self._handler = Handler

    def _send_health(self, handler: BaseHTTPRequestHandler) -> None:
        with self._spec_lock:
            dynamic = len(self._specs)
        handler.send_response(200)
        payload = json.dumps(
            {"ok": True, "tools": len(self.registry.names()), "dynamic": dynamic},
            separators=(",", ":"),
        ).encode("utf-8")
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    def _register(self, handler: BaseHTTPRequestHandler, payload: Any) -> None:
        if not self.allow_dynamic:
            handler._send(403, {"error": "dynamic registration disabled"})
            return
        try:
            name = str(payload["name"])
            endpoint = str(payload["endpoint"])
            parameters = payload["parameters"]
            if not isinstance(parameters, dict):
                raise ValueError("parameters must be a JSON-Schema object")
            method = str(payload.get("method", "POST")).upper()
            if method not in ("GET", "POST"):
                raise ValueError("method must be GET or POST")
            # Deep validation (SSRF, allowlist, redirects) is done in Rust on
            # every invocation; this is just early feedback for callers.
            if urlsplit(endpoint).scheme not in ("http", "https"):
                raise NetworkToolError("unsupported endpoint scheme")
            if len(name) > 128 or len(name) < 1:
                raise ValueError("tool name must be 1..128 characters")
            if name in self.registry.names():
                raise ValueError(f"tool already registered: {name}")
            raw_headers = dict(payload.get("headers", {}))
            if len(raw_headers) > 32:
                raise ValueError("too many headers (max 32)")
            headers = {str(k): str(v) for k, v in raw_headers.items()}
            if any(len(key) > 128 or len(value) > 4096 for key, value in headers.items()):
                raise ValueError("header key/value too large")
            spec = _RemoteSpec(
                name=name,
                endpoint=endpoint,
                method=method,
                headers=headers,
                allow_private=self.allow_private_endpoints,
            )
            tool = Tool(
                name=name,
                description=str(payload.get("description", "Dynamically registered HTTP tool")),
                parameters=parameters,
                handler=lambda args, s=spec: self._invoke_remote(s, args),
                timeout_secs=15.0,
                max_result_chars=1 << 16,
            )
            self.registry.register(tool)
        except (KeyError, ValueError, NetworkToolError, ToolError) as exc:
            handler._send(400, {"error": str(exc)})
            return
        with self._spec_lock:
            self._specs[name] = spec
        if self.telemetry is not None:
            self.telemetry.emit("tool_proxy_register", name=name)
        handler._send(200, {"ok": True, "name": name})

    def _invoke_remote(self, spec: _RemoteSpec, arguments: dict[str, Any]) -> str:
        response = http_request(
            spec.method,
            spec.endpoint,
            body=arguments if spec.method == "POST" else None,
            headers=spec.headers,
            timeout_secs=15.0,
            max_bytes=1 << 16,
            allow_private=spec.allow_private,
        )
        return f"status={response.status} url={response.url}\n{response.text}"

    @staticmethod
    def _parse_calls(payload: Any) -> list[ParsedToolCall]:
        items = payload.get("calls")
        if not isinstance(items, list):
            raise ValueError("payload needs a 'calls' list")
        calls: list[ParsedToolCall] = []
        for item in items:
            name = str(item["name"])
            arguments = dict(item.get("arguments", {}))
            calls.append(
                ParsedToolCall(
                    id=str(item.get("id", "")),
                    name=name,
                    arguments=arguments,
                    arguments_json=json.dumps(arguments, separators=(",", ":")),
                )
            )
        return calls

    def _call_batch(self, handler: BaseHTTPRequestHandler, payload: Any) -> None:
        try:
            calls = self._parse_calls(payload)
        except (KeyError, ValueError, TypeError) as exc:
            handler._send(400, {"error": str(exc)})
            return
        self._execute(handler, calls)

    def _call_one(self, handler: BaseHTTPRequestHandler, name: str, payload: Any) -> None:
        arguments = dict(payload.get("arguments", payload))
        call = ParsedToolCall(
            id=str(payload.get("id", "")),
            name=name,
            arguments=arguments,
            arguments_json=json.dumps(arguments, separators=(",", ":")),
        )
        self._execute(handler, [call])

    def _execute(self, handler: BaseHTTPRequestHandler, calls: Sequence[ParsedToolCall]) -> None:
        try:
            results = self.registry.execute(calls)
        except ToolError as exc:
            handler._send(400, {"error": str(exc)})
            return
        handler._send(
            200,
            {
                "results": [
                    dataclasses.asdict(result) for result in results
                ]
            },
        )

    def start(self) -> int:
        """Start the server on a daemon thread and return the bound port."""
        if self._server is not None:
            raise NetworkToolError("proxy already started")
        server = ThreadingHTTPServer((self.host, self.port), self._handler)
        server.daemon_threads = True
        self._server = server
        self.port = server.server_address[1]
        self._thread = threading.Thread(
            target=server.serve_forever, name="megarl-tool-proxy", daemon=True
        )
        self._thread.start()
        return self.port

    def stop(self) -> None:
        server, thread = self._server, self._thread
        self._server, self._thread = None, None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5.0)

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> ToolHTTPProxy:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = [
    "CallbackSearchBackend",
    "HTTPResponse",
    "HTTPSearchBackend",
    "NetworkToolError",
    "SearchBackend",
    "ToolHTTPProxy",
    "default_network_registry",
    "http_request",
    "make_fetch_tool",
    "make_search_tool",
]
