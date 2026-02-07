# Copyright (c) 2026 The neuraLQX Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Authenticator utility for signing and verifying neuraLQX HTML output logs.

Supports both:
- Legacy format: <p id="rd">..., <p id="sgn">..., <p id="pk">...
- New UI format: <script id="nqx-data" type="application/json">... plus <p id="sgn"> and <p id="pk">

The recommended API is:
  Authenticator.verify_file(path) -> (ok: bool, payload_str: str, error: str|None)

A legacy-compatible verify_signature(path) is kept, which prints results.
"""

from __future__ import annotations

import base64
import json
import secrets as s
import uuid as u
from dataclasses import dataclass
from typing import Optional, Tuple

from html import unescape as _html_unescape
from bs4 import BeautifulSoup
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    payload: str
    signature_b64: str
    public_key_pem: str
    error: Optional[str] = None
    format: str = "unknown"  # "legacy" or "nqx"


class Authenticator:

    @staticmethod
    def generate_key_pair():
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()
        pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")
        return private_key, public_key, pem

    @staticmethod
    def sign_data(private_key, data: str) -> str:
        if not isinstance(data, str):
            data = str(data)

        signature = private_key.sign(
            data.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    @staticmethod
    def _read_file_text(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def extract_from_html(html_text: str) -> Tuple[str, str, str, str]:
        soup = BeautifulSoup(html_text, features="html.parser")

        sgn_tag = soup.find("p", id="sgn")
        pk_tag = soup.find("p", id="pk")
        if sgn_tag is None or pk_tag is None:
            raise ValueError("Missing required <p id='sgn'> and/or <p id='pk'> fields.")

        signature_b64 = sgn_tag.get_text(strip=True)

        # normalise PEM
        public_key_pem = pk_tag.get_text()
        public_key_pem = public_key_pem.strip() + "\n"

        # prefer new format
        script_tag = soup.find(
            "script", id="nqx-data", attrs={"type": "application/json"}
        )
        if script_tag is not None:
            payload_raw = script_tag.string
            if payload_raw is None:
                payload_raw = script_tag.get_text()
            payload_raw = payload_raw.strip()
            if not payload_raw:
                raise ValueError("Empty nqx-data JSON payload.")
            try:
                json.loads(payload_raw)
            except Exception as e:
                raise ValueError(f"Malformed JSON in nqx-data payload: {e}") from e
            return payload_raw, signature_b64, public_key_pem, "nqx"

        # legacy payload
        rd_tag = soup.find("p", id="rd")
        if rd_tag is None:
            raise ValueError(
                "Missing payload: expected <script id='nqx-data'> or <p id='rd'>."
            )

        # use inner HTML to avoid losing "<unsafe>" which HTML parser treats as a tag
        # then unescape HTML entities back to the original payload
        inner = rd_tag.decode_contents()
        payload = _html_unescape(inner).strip()

        if payload == "":
            raise ValueError("Empty legacy payload in <p id='rd'>.")
        return payload, signature_b64, public_key_pem, "legacy"

    @staticmethod
    def verify_payload(
        payload: str, signature_b64: str, public_key_pem: str
    ) -> VerificationResult:
        try:
            public_key = serialization.load_pem_public_key(
                public_key_pem.encode("utf-8")
            )
        except Exception as e:
            return VerificationResult(
                ok=False,
                payload=payload,
                signature_b64=signature_b64,
                public_key_pem=public_key_pem,
                error=f"Invalid public key PEM: {e}",
            )

        try:
            sig = base64.b64decode(signature_b64.encode("utf-8"))
        except Exception as e:
            return VerificationResult(
                ok=False,
                payload=payload,
                signature_b64=signature_b64,
                public_key_pem=public_key_pem,
                error=f"Invalid base64 signature: {e}",
            )

        try:
            public_key.verify(
                sig,
                payload.encode("utf-8"),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return VerificationResult(
                ok=True,
                payload=payload,
                signature_b64=signature_b64,
                public_key_pem=public_key_pem,
                error=None,
            )
        except Exception as e:
            return VerificationResult(
                ok=False,
                payload=payload,
                signature_b64=signature_b64,
                public_key_pem=public_key_pem,
                error=f"Signature verification failed: {e}",
            )

    @staticmethod
    def verify_file(path: str) -> VerificationResult:
        html_text = Authenticator._read_file_text(path)
        payload, sig, pk, fmt = Authenticator.extract_from_html(html_text)
        res = Authenticator.verify_payload(payload, sig, pk)
        return VerificationResult(
            ok=res.ok,
            payload=res.payload,
            signature_b64=res.signature_b64,
            public_key_pem=res.public_key_pem,
            error=res.error,
            format=fmt,
        )

    @staticmethod
    def verify_signature(path: str) -> None:
        try:
            res = Authenticator.verify_file(path)
        except FileNotFoundError:
            raise
        except Exception as e:
            print(f"\n\nSignature verification failed\n(Reason: {e})\n\n")
            return

        if res.ok:
            print("Validated.")
        else:
            print(f"\n\nSignature verification failed\n(Reason: {res.error})\n\n")


def get_hash():  # pylint: disable=C0115,C0116
    """
    Generate a short unique identifier suitable for tagging outputs.

    The identifier is built from:
    - a 6-hex-character prefix derived from a UUID4, and
    - a 10-hex-character cryptographic token (5 bytes) from ``secrets.token_hex``.

    The two components are joined with a hyphen, e.g. ``"a1b2c3-7f3a9c12de"``.

    :returns: A short unique string identifier.
    """

    return f"{u.uuid4().hex[:6]}-{s.token_hex(nbytes=5)}"
