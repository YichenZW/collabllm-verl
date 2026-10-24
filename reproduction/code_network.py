"""HTTP compatibility for the benchmark's original public ZIP fixtures.

The host rejects Python's default User-Agent with HTTP 403. Keep the original
URLs, bytes, solutions and assertions; only configure default HTTP headers for
tests which depend on this host. Explicit candidate-supplied headers still win.
"""
import urllib.request
from urllib.parse import urlsplit


class FixtureUserAgent(urllib.request.BaseHandler):
    handler_order = 490  # Set a host-specific default before urllib's defaults.

    def http_request(self, request):
        if urlsplit(request.full_url).hostname == "getsamplefiles.com" and not request.has_header("User-agent"):
            request.add_unredirected_header("User-agent", "Mozilla/5.0")
        return request

    https_request = http_request


def configure_fixture_http(test_code):
    if "https://getsamplefiles.com/download/zip/" not in test_code:
        return None
    urllib.request.install_opener(urllib.request.build_opener(FixtureUserAgent()))
    # Preserve other hosts and all explicit candidate headers. In particular,
    # the benchmark's invalid-URL test must retain its original HTTP behavior.
    import requests
    original = requests.Session.prepare_request
    if not getattr(original, "_fixture_ua_compat", False):
        def prepare(session, request):
            result = original(session, request)
            explicit = any(key.lower() == "user-agent" for key in (request.headers or {}))
            if (urlsplit(result.url).hostname == "getsamplefiles.com" and not explicit
                    and session.headers.get("User-Agent") == requests.utils.default_user_agent()):
                result.headers["User-Agent"] = "Mozilla/5.0"
            return result
        prepare._fixture_ua_compat = True
        requests.Session.prepare_request = prepare
    return "getsamplefiles-browser-user-agent-v2"
