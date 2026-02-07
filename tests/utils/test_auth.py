#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  Licensed under the Apache License 2.0, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import json
import pytest

from neuralqx.utils.misc.auth import Authenticator


def _make_legacy_html(payload: str, sig: str, pem: str) -> str:
    return f"""
    <html><body>
      <div id="hidden-data">
        <p id="sgn">{sig}</p>
        <p id="rd">{payload}</p>
        <p id="pk">{pem}</p>
      </div>
    </body></html>
    """


def _make_nqx_html(payload_json_str: str, sig: str, pem: str) -> str:
    # mimic the output: payload in nqx-data script tag
    return f"""
    <html><body>
      <script id="nqx-data" type="application/json">
{payload_json_str}
      </script>
      <div id="hidden-data">
        <p id="sgn">{sig}</p>
        <p id="pk">{pem}</p>
      </div>
    </body></html>
    """


def test_generate_key_pair_returns_pem_string():
    prk, puk, pem = Authenticator.generate_key_pair()
    assert prk is not None
    assert puk is not None
    assert isinstance(pem, str)
    assert "BEGIN PUBLIC KEY" in pem


def test_sign_and_verify_payload_roundtrip_ok():
    prk, _, pem = Authenticator.generate_key_pair()
    payload = "hello world"
    sig = Authenticator.sign_data(prk, payload)

    res = Authenticator.verify_payload(payload, sig, pem)
    assert res.ok is True
    assert res.error is None


def test_verify_payload_detects_tampering():
    prk, _, pem = Authenticator.generate_key_pair()
    payload = "original"
    sig = Authenticator.sign_data(prk, payload)

    tampered = "original BUT CHANGED"
    res = Authenticator.verify_payload(tampered, sig, pem)
    assert res.ok is False
    assert res.error is not None
    assert "Signature verification failed" in res.error


def test_extract_from_html_legacy_ok():
    prk, _, pem = Authenticator.generate_key_pair()
    payload = "raw<unsafe>&"
    escaped_payload = "raw&lt;unsafe&gt;&amp;"
    sig = Authenticator.sign_data(prk, payload)

    html = _make_legacy_html(escaped_payload, sig, pem)
    p, s, k, fmt = Authenticator.extract_from_html(html)

    assert fmt == "legacy"
    assert p == payload
    assert s == sig
    assert k == pem


def test_extract_from_html_nqx_ok_and_json_validated():
    prk, _, pem = Authenticator.generate_key_pair()
    payload_obj = {"Sec": {"k": "v<unsafe>"}}
    payload_json = json.dumps(payload_obj)

    sig = Authenticator.sign_data(prk, payload_json)
    html = _make_nqx_html(payload_json, sig, pem)

    p, s, k, fmt = Authenticator.extract_from_html(html)
    assert fmt == "nqx"
    assert json.loads(p) == payload_obj
    assert s == sig
    assert k == pem


def test_verify_file_nqx_ok(tmp_path):
    prk, _, pem = Authenticator.generate_key_pair()
    payload_obj = {"A": {"x": 1}}
    payload_json = json.dumps(payload_obj)
    sig = Authenticator.sign_data(prk, payload_json)

    html = _make_nqx_html(payload_json, sig, pem)
    f = tmp_path / "out.html"
    f.write_text(html, encoding="utf-8")

    res = Authenticator.verify_file(str(f))
    assert res.ok is True
    assert res.format == "nqx"
    assert json.loads(res.payload) == payload_obj


def test_verify_file_legacy_ok(tmp_path):
    prk, _, pem = Authenticator.generate_key_pair()
    payload = "SOME RAW DATA"
    sig = Authenticator.sign_data(prk, payload)

    html = _make_legacy_html(payload, sig, pem)
    f = tmp_path / "out.html"
    f.write_text(html, encoding="utf-8")

    res = Authenticator.verify_file(str(f))
    assert res.ok is True
    assert res.format == "legacy"
    assert res.payload == payload


def test_verify_file_missing_fields_raises_value_error(tmp_path):
    html = "<html><body><p id='rd'>x</p></body></html>"
    f = tmp_path / "out.html"
    f.write_text(html, encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required"):
        Authenticator.verify_file(str(f))


def test_verify_file_missing_payload_raises_value_error(tmp_path):
    prk, _, pem = Authenticator.generate_key_pair()
    sig = Authenticator.sign_data(prk, "anything")

    html = f"<html><body><p id='sgn'>{sig}</p><p id='pk'>{pem}</p></body></html>"
    f = tmp_path / "out.html"
    f.write_text(html, encoding="utf-8")

    with pytest.raises(ValueError, match="Missing payload"):
        Authenticator.verify_file(str(f))


def test_extract_from_html_nqx_malformed_json_raises():
    prk, _, pem = Authenticator.generate_key_pair()
    bad_json = "{not: json}"
    sig = Authenticator.sign_data(prk, bad_json)
    html = _make_nqx_html(bad_json, sig, pem)

    with pytest.raises(ValueError, match="Malformed JSON"):
        Authenticator.extract_from_html(html)


def test_verify_payload_invalid_pem_returns_not_ok():
    prk, _, _pem = Authenticator.generate_key_pair()
    payload = "abc"
    sig = Authenticator.sign_data(prk, payload)

    res = Authenticator.verify_payload(payload, sig, "NOT A PEM")
    assert res.ok is False
    assert res.error is not None
    assert "Invalid public key PEM" in res.error


def test_verify_payload_invalid_signature_base64_returns_not_ok():
    prk, _, pem = Authenticator.generate_key_pair()
    payload = "abc"
    sig = "%%%NOT-BASE64%%%"

    res = Authenticator.verify_payload(payload, sig, pem)
    assert res.ok is False
    assert "Invalid base64 signature" in res.error


def test_verify_file_missing_file_raises_filenotfound(tmp_path):
    missing = tmp_path / "nope.html"
    with pytest.raises(FileNotFoundError):
        Authenticator.verify_file(str(missing))


def test_verify_signature_prints_success(capsys, tmp_path):
    prk, _, pem = Authenticator.generate_key_pair()
    payload_obj = {"ok": True}
    payload_json = json.dumps(payload_obj)
    sig = Authenticator.sign_data(prk, payload_json)
    html = _make_nqx_html(payload_json, sig, pem)

    f = tmp_path / "out.html"
    f.write_text(html, encoding="utf-8")

    Authenticator.verify_signature(str(f))
    out = capsys.readouterr().out
    assert "validated." in out.lower()


def test_verify_signature_prints_failure(capsys, tmp_path):
    prk, _, pem = Authenticator.generate_key_pair()
    payload = json.dumps({"x": 1})
    sig = Authenticator.sign_data(prk, payload)

    # tamper payload but keep signature
    tampered_payload = json.dumps({"x": 2})
    html = _make_nqx_html(tampered_payload, sig, pem)

    f = tmp_path / "out.html"
    f.write_text(html, encoding="utf-8")

    Authenticator.verify_signature(str(f))
    out = capsys.readouterr().out
    assert "verification failed" in out.lower()
