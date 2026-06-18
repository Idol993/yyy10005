import json
import base64
from typing import Dict, Optional, Tuple

from .curve import CurvePoint, Secp256r1, is_point_on_curve, validate_point, scalar_mult_base
from .ecdsa import _validate_private_key, private_key_to_hex, public_key_to_hex


JWK_CURVE_P256 = "P-256"
JWK_KTY_EC = "EC"
JWK_CRV = "crv"
JWK_KTY = "kty"
JWK_X = "x"
JWK_Y = "y"
JWK_D = "d"


def _base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _base64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def private_key_to_jwk(private_key: int, public_key: CurvePoint = None) -> Dict:
    _validate_private_key(private_key)

    if public_key is None:
        public_key = scalar_mult_base(private_key)
    else:
        validate_point(public_key)
        expected_pub = scalar_mult_base(private_key)
        if expected_pub != public_key:
            raise ValueError("Provided public key does not match private key")

    byte_len = 32
    d_bytes = private_key.to_bytes(byte_len, byteorder="big")
    x_bytes = public_key.x.to_bytes(byte_len, byteorder="big")
    y_bytes = public_key.y.to_bytes(byte_len, byteorder="big")

    return {
        JWK_KTY: JWK_KTY_EC,
        JWK_CRV: JWK_CURVE_P256,
        JWK_X: _base64url_encode(x_bytes),
        JWK_Y: _base64url_encode(y_bytes),
        JWK_D: _base64url_encode(d_bytes),
    }


def public_key_to_jwk(public_key: CurvePoint) -> Dict:
    validate_point(public_key)

    byte_len = 32
    x_bytes = public_key.x.to_bytes(byte_len, byteorder="big")
    y_bytes = public_key.y.to_bytes(byte_len, byteorder="big")

    return {
        JWK_KTY: JWK_KTY_EC,
        JWK_CRV: JWK_CURVE_P256,
        JWK_X: _base64url_encode(x_bytes),
        JWK_Y: _base64url_encode(y_bytes),
    }


def jwk_to_private_key(jwk: Dict) -> Tuple[int, CurvePoint]:
    if not isinstance(jwk, dict):
        raise TypeError("JWK must be a dictionary")

    if jwk.get(JWK_KTY) != JWK_KTY_EC:
        raise ValueError(f"Invalid kty: expected 'EC', got {jwk.get(JWK_KTY)!r}")

    if jwk.get(JWK_CRV) != JWK_CURVE_P256:
        raise ValueError(f"Invalid crv: expected 'P-256', got {jwk.get(JWK_CRV)!r}")

    if JWK_D not in jwk:
        raise ValueError("JWK does not contain private key (missing 'd' field)")

    for field in (JWK_X, JWK_Y, JWK_D):
        if not isinstance(jwk.get(field), str):
            raise ValueError(f"JWK field '{field}' must be a string")

    try:
        x_bytes = _base64url_decode(jwk[JWK_X])
        y_bytes = _base64url_decode(jwk[JWK_Y])
        d_bytes = _base64url_decode(jwk[JWK_D])
    except Exception as e:
        raise ValueError(f"Invalid base64url encoding in JWK: {e}")

    if len(x_bytes) != 32:
        raise ValueError(f"Invalid x coordinate length: {len(x_bytes)} bytes, expected 32")
    if len(y_bytes) != 32:
        raise ValueError(f"Invalid y coordinate length: {len(y_bytes)} bytes, expected 32")
    if len(d_bytes) != 32:
        raise ValueError(f"Invalid private key length: {len(d_bytes)} bytes, expected 32")

    x = int.from_bytes(x_bytes, byteorder="big")
    y = int.from_bytes(y_bytes, byteorder="big")
    private_key = int.from_bytes(d_bytes, byteorder="big")

    if x < 0 or x >= Secp256r1.p:
        raise ValueError("x coordinate out of field range")
    if y < 0 or y >= Secp256r1.p:
        raise ValueError("y coordinate out of field range")

    _validate_private_key(private_key)

    public_key = CurvePoint(x, y, False)
    if not is_point_on_curve(public_key):
        raise ValueError("JWK public key point is not on secp256r1 curve")

    expected_pub = scalar_mult_base(private_key)
    if expected_pub != public_key:
        raise ValueError("JWK private key does not match public key")

    return private_key, public_key


def jwk_to_public_key(jwk: Dict) -> CurvePoint:
    if not isinstance(jwk, dict):
        raise TypeError("JWK must be a dictionary")

    if jwk.get(JWK_KTY) != JWK_KTY_EC:
        raise ValueError(f"Invalid kty: expected 'EC', got {jwk.get(JWK_KTY)!r}")

    if jwk.get(JWK_CRV) != JWK_CURVE_P256:
        raise ValueError(f"Invalid crv: expected 'P-256', got {jwk.get(JWK_CRV)!r}")

    for field in (JWK_X, JWK_Y):
        if not isinstance(jwk.get(field), str):
            raise ValueError(f"JWK field '{field}' must be a string")

    try:
        x_bytes = _base64url_decode(jwk[JWK_X])
        y_bytes = _base64url_decode(jwk[JWK_Y])
    except Exception as e:
        raise ValueError(f"Invalid base64url encoding in JWK: {e}")

    if len(x_bytes) != 32:
        raise ValueError(f"Invalid x coordinate length: {len(x_bytes)} bytes, expected 32")
    if len(y_bytes) != 32:
        raise ValueError(f"Invalid y coordinate length: {len(y_bytes)} bytes, expected 32")

    x = int.from_bytes(x_bytes, byteorder="big")
    y = int.from_bytes(y_bytes, byteorder="big")

    if x < 0 or x >= Secp256r1.p:
        raise ValueError("x coordinate out of field range")
    if y < 0 or y >= Secp256r1.p:
        raise ValueError("y coordinate out of field range")

    public_key = CurvePoint(x, y, False)
    if not is_point_on_curve(public_key):
        raise ValueError("JWK public key point is not on secp256r1 curve")

    return public_key


def jwk_to_json(jwk: Dict, pretty: bool = False) -> str:
    if not isinstance(jwk, dict):
        raise TypeError("JWK must be a dictionary")
    if pretty:
        return json.dumps(jwk, indent=2, sort_keys=True)
    return json.dumps(jwk, sort_keys=True, separators=(",", ":"))


def json_to_jwk(json_str: str) -> Dict:
    if not isinstance(json_str, str):
        raise TypeError("JSON string must be a string")
    try:
        jwk = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    if not isinstance(jwk, dict):
        raise ValueError("JWK must be a JSON object")
    return jwk
