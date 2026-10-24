import urllib.request

import requests

from reproduction.code_network import configure_fixture_http


def test_fixture_header_preserves_other_hosts_and_explicit_candidate_headers(monkeypatch):
    monkeypatch.setattr(urllib.request, "_opener", urllib.request._opener)
    monkeypatch.setattr(requests.Session, "prepare_request", requests.Session.prepare_request)
    assert configure_fixture_http("unrelated benchmark") is None
    configure_fixture_http("https://getsamplefiles.com/download/zip/sample-1.zip")
    for host, expected in [("getsamplefiles.com", "Mozilla/5.0"),
                           ("invalidurl.com", "Python-urllib/3")]:
        req = urllib.request.Request(f"https://{host}/file.zip")
        for handler in urllib.request._opener.process_request["https"]:
            req = handler.https_request(req)
        assert req.get_header("User-agent").startswith(expected)
        with requests.Session() as session:
            prepared = session.prepare_request(requests.Request("GET", f"https://{host}/file.zip"))
            assert prepared.headers["User-Agent"] == (
                "Mozilla/5.0" if host == "getsamplefiles.com" else requests.utils.default_user_agent())
            explicit = session.prepare_request(requests.Request("GET", f"https://{host}/file.zip",
                                                headers={"User-Agent": "candidate-specific"}))
            assert explicit.headers["User-Agent"] == "candidate-specific"
        req = urllib.request.Request(f"https://{host}/file.zip", headers={"User-Agent": "candidate-specific"})
        for handler in urllib.request._opener.process_request["https"]:
            req = handler.https_request(req)
        assert req.get_header("User-agent") == "candidate-specific"
