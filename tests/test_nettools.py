"""Network tool layer: hardened fetch/search and the agent tool proxy."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from megarl.nettools import (
    CallbackSearchBackend,
    HTTPSearchBackend,
    NetworkToolError,
    ToolHTTPProxy,
    default_network_registry,
    http_request,
    make_fetch_tool,
    make_search_tool,
)
from megarl.tools import ParsedToolCall, ToolRegistry


class _FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _reply(self, status, body, content_type="text/plain"):
        payload = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/ok":
            self._reply(200, "hello network")
        elif self.path == "/large":
            self._reply(200, "x" * 1000)
        elif self.path == "/binary":
            self._reply(200, b"\x00\x01\x02", "application/octet-stream")
        elif self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/ok")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/evil-redirect":
            self.send_response(302)
            self.send_header("Location", "file:///etc/passwd")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/search":
            self._reply(
                200,
                json.dumps(
                    {"results": [{"title": "A", "url": "https://a", "content": "snippet"}]}
                ),
                "application/json",
            )
        else:
            self._reply(404, "not found")

    def do_POST(self):
        size = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(size) or b"{}")
        if self.path == "/echo":
            self._reply(200, json.dumps({"echo": body}), "application/json")
        elif self.path == "/search":
            self._reply(
                200,
                json.dumps(
                    {
                        "results": [
                            {
                                "title": f"result for {body.get('query')}",
                                "url": "https://example.com",
                                "content": "body",
                            }
                        ]
                    }
                ),
                "application/json",
            )
        else:
            self._reply(404, "not found")

    def log_message(self, *args):
        pass


@pytest.fixture
def local_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FixtureHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Safe HTTP core / fetch tool
# ---------------------------------------------------------------------------


def test_http_request_text_and_content_type(local_server):
    response = http_request("GET", local_server + "/ok", allow_private=True)
    assert response.status == 200
    assert response.text == "hello network"
    assert response.content_type.startswith("text/plain")


def test_http_request_follows_validated_redirects(local_server):
    response = http_request("GET", local_server + "/redirect", allow_private=True)
    assert response.status == 200
    assert response.text == "hello network"
    assert response.url.endswith("/ok")


def test_redirect_target_is_revalidated(local_server):
    with pytest.raises(NetworkToolError, match="unsupported URL scheme"):
        http_request("GET", local_server + "/evil-redirect", allow_private=True)


def test_fetch_tool_blocks_private_hosts_and_userinfo(local_server):
    fetch = make_fetch_tool()  # public-only by default
    with pytest.raises(NetworkToolError):
        fetch.invoke({"url": local_server + "/ok"})
    with pytest.raises(NetworkToolError, match="userinfo"):
        fetch.invoke({"url": "http://user:pass@127.0.0.1/x"})
    with pytest.raises(NetworkToolError, match="scheme"):
        fetch.invoke({"url": "file:///etc/passwd"})


def test_fetch_tool_size_cap_and_binary_filter(local_server):
    fetch = make_fetch_tool(allow_private=True)
    limited = fetch.invoke({"url": local_server + "/large", "max_bytes": 100})
    assert "truncated" in limited
    assert len(limited.split("\n", 1)[1]) <= 100
    binary = fetch.invoke({"url": local_server + "/binary"})
    assert "binary content omitted" in binary


def test_fetch_failure_is_a_structured_tool_result(local_server):
    registry = ToolRegistry([make_fetch_tool()])
    result = registry.execute(
        [ParsedToolCall("f", "fetch", {"url": local_server + "/ok"}, '{"url":"' + local_server + '/ok"}')]
    )[0]
    assert result.ok is False and "non-public address" in result.error
    registry.close()


# ---------------------------------------------------------------------------
# Search tools
# ---------------------------------------------------------------------------


def test_callback_search_backend():
    tool = make_search_tool(CallbackSearchBackend(lambda q, n: [{"title": q, "url": "u", "content": "c"}]))
    out = tool.invoke({"query": "cats", "limit": 2})
    assert "cats" in out and "[1]" in out


def test_unconfigured_search_fails_loudly():
    with pytest.raises(NetworkToolError, match="no search backend"):
        make_search_tool(None).invoke({"query": "x"})


def test_http_search_backend(local_server):
    backend = HTTPSearchBackend(local_server + "/search", allow_private=True)
    results = backend.search("quantum", 3)
    assert results == [
        {"title": "result for quantum", "url": "https://example.com", "content": "body"}
    ]
    registry = default_network_registry(search_backend=backend, allow_private=True)
    assert set(registry.names()) == {"fetch", "search"}
    registry.close()


# ---------------------------------------------------------------------------
# Tool proxy
# ---------------------------------------------------------------------------


def _request(url, *, method="GET", payload=None, headers=None):
    from urllib.request import Request, urlopen

    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(url, data=data, method=method, headers=headers or {})
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urlopen(request, timeout=10) as response:
        return response.status, json.loads(response.read())


def test_proxy_health_discovery_and_calls():
    registry = ToolRegistry([make_search_tool(None)])
    with ToolHTTPProxy(registry) as proxy:
        base = f"http://127.0.0.1:{proxy.port}"
        status, health = _request(base + "/health")
        assert status == 200 and health["ok"] is True
        status, tools = _request(base + "/tools")
        assert status == 200 and tools["tools"][0]["function"]["name"] == "search"

        status, single = _request(
            base + "/tools/search/call",
            method="POST",
            payload={"arguments": {"query": "x"}},
        )
        assert status == 200 and single["results"][0]["ok"] is False  # no backend
    registry.close()


def test_proxy_auth_and_body_limit():
    registry = ToolRegistry()
    with ToolHTTPProxy(registry, token="secret", max_body_bytes=64) as proxy:
        base = f"http://127.0.0.1:{proxy.port}"
        status, health = _request(base + "/health")  # health stays open
        assert status == 200 and health["ok"] is True
        with pytest.raises(Exception) as exc:
            _request(base + "/tools")
        assert exc.value.code == 401
        with pytest.raises(Exception) as exc:
            _request(
                base + "/tools",
                method="POST",
                payload={"name": "x" * 200},
                headers={"Authorization": "Bearer secret"},
            )
        assert exc.value.code == 400  # body cap
    registry.close()


def test_proxy_auth_enforced():
    registry = ToolRegistry()
    with ToolHTTPProxy(registry, token="secret") as proxy:
        base = f"http://127.0.0.1:{proxy.port}"
        with pytest.raises(Exception) as exc:
            _request(base + "/tools")
        assert exc.value.code == 401
        status, tools = _request(
            base + "/tools", headers={"Authorization": "Bearer secret"}
        )
        assert status == 200 and tools["tools"] == []
    registry.close()


def test_proxy_dynamic_http_handler(local_server):
    registry = ToolRegistry()
    with ToolHTTPProxy(registry, allow_private_endpoints=True) as proxy:
        base = f"http://127.0.0.1:{proxy.port}"
        status, registered = _request(
            base + "/tools",
            method="POST",
            payload={
                "name": "echoer",
                "endpoint": local_server + "/echo",
                "description": "echo arguments",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "additionalProperties": False,
                },
            },
        )
        assert status == 200 and registered["ok"] is True
        status, result = _request(
            base + "/tools/echoer/call",
            method="POST",
            payload={"arguments": {"x": 7}},
        )
        assert status == 200
        assert result["results"][0]["ok"] is True
        assert '"x": 7' in result["results"][0]["content"]
        status, batch = _request(
            base + "/tools/call",
            method="POST",
            payload={"calls": [{"name": "echoer", "arguments": {"x": 1}, "id": "a"}]},
        )
        assert status == 200 and batch["results"][0]["call_id"] == "a"
        status, deleted = _request(base + "/tools/echoer", method="DELETE")
        assert status == 200 and deleted["ok"] is True
        assert "echoer" not in registry.names()
    registry.close()


def test_proxy_all_or_nothing_batch():
    registry = ToolRegistry([make_search_tool(None)])
    with ToolHTTPProxy(registry) as proxy:
        base = f"http://127.0.0.1:{proxy.port}"
        with pytest.raises(Exception) as exc:
            _request(
                base + "/tools/call",
                method="POST",
                payload={"calls": [{"name": "search", "arguments": {}}, {"name": "nope", "arguments": {}}]},
            )
        assert exc.value.code == 400  # validation fails before any execution
    registry.close()
