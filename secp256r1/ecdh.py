from typing import Optional, Tuple

from .curve import (
    CurvePoint,
    Secp256r1,
    INFINITY,
    is_point_on_curve,
    validate_point,
    scalar_mult,
)
from .field_ops import mod_mul, sha256, hmac_sha256


HKDF_SHA256_MAX_LENGTH = 255 * 32


def _validate_ecdh_public_key(public_key: CurvePoint, curve=Secp256r1) -> None:
    validate_point(public_key)

    if public_key.x == 0 and public_key.y == 0:
        raise ValueError("ECDH public key cannot be (0, 0)")

    if not is_point_on_curve(public_key, curve):
        raise ValueError("ECDH public key is not on the curve")

    nP = scalar_mult(curve.n, public_key, curve)
    if not nP.is_infinity():
        raise ValueError("ECDH public key is not in the correct prime-order subgroup")


def ecdh_compute_shared_secret(
    private_key: int,
    peer_public_key: CurvePoint,
    cofactor: bool = True,
    curve=Secp256r1,
) -> bytes:
    if not isinstance(private_key, int):
        raise TypeError("Private key must be an integer")
    if private_key <= 0 or private_key >= curve.n:
        raise ValueError("Private key must be in range [1, n-1]")

    _validate_ecdh_public_key(peer_public_key, curve)

    scalar = private_key
    if cofactor and curve.h != 1:
        scalar = mod_mul(private_key, curve.h, curve.n)

    shared_point = scalar_mult(scalar, peer_public_key, curve)

    if shared_point.is_infinity():
        raise ValueError("ECDH computation resulted in point at infinity (invalid shared secret)")

    byte_len = (curve.p.bit_length() + 7) // 8
    return shared_point.x.to_bytes(byte_len, byteorder="big")


def ecdh_compute_shared_secret_and_hash(
    private_key: int,
    peer_public_key: CurvePoint,
    cofactor: bool = True,
    curve=Secp256r1,
) -> bytes:
    raw_secret = ecdh_compute_shared_secret(private_key, peer_public_key, cofactor, curve)
    return sha256(raw_secret)


def hkdf_sha256_extract(salt: Optional[bytes], ikm: bytes) -> bytes:
    if not isinstance(ikm, (bytes, bytearray)):
        raise TypeError("IKM (input keying material) must be bytes")
    if salt is None:
        salt = b"\x00" * 32
    elif not isinstance(salt, (bytes, bytearray)):
        raise TypeError("Salt must be bytes or None")
    return hmac_sha256(salt, ikm)


def hkdf_sha256_expand(prk: bytes, info: bytes, length: int) -> bytes:
    if not isinstance(prk, (bytes, bytearray)):
        raise TypeError("PRK must be bytes")
    if not isinstance(info, (bytes, bytearray)):
        raise TypeError("Info must be bytes")
    if not isinstance(length, int) or length < 1:
        raise ValueError("Length must be a positive integer")
    if length > HKDF_SHA256_MAX_LENGTH:
        raise ValueError(f"HKDF-SHA256 output length cannot exceed {HKDF_SHA256_MAX_LENGTH} bytes")
    if len(prk) < 32:
        raise ValueError(f"PRK must be at least 32 bytes (SHA-256 output size), got {len(prk)}")

    n = (length + 31) // 32
    okm = b""
    t = b""
    for i in range(1, n + 1):
        t = hmac_sha256(prk, t + info + bytes([i]))
        okm += t
    return okm[:length]


def hkdf_sha256(
    ikm: bytes,
    salt: Optional[bytes] = None,
    info: bytes = b"",
    length: int = 32,
) -> bytes:
    if not isinstance(ikm, (bytes, bytearray)):
        raise TypeError("IKM must be bytes")
    if not isinstance(length, int) or length < 1:
        raise ValueError("Length must be a positive integer")
    if length > HKDF_SHA256_MAX_LENGTH:
        raise ValueError(f"HKDF-SHA256 output length cannot exceed {HKDF_SHA256_MAX_LENGTH} bytes")
    prk = hkdf_sha256_extract(salt, ikm)
    return hkdf_sha256_expand(prk, info, length)


def ecdh_derive_key(
    private_key: int,
    peer_public_key: CurvePoint,
    kdf: str = "sha256",
    salt: Optional[bytes] = None,
    info: bytes = b"",
    length: int = 32,
    cofactor: bool = True,
    curve=Secp256r1,
) -> bytes:
    raw_secret = ecdh_compute_shared_secret(private_key, peer_public_key, cofactor, curve)

    if kdf == "sha256":
        if length != 32:
            raise ValueError("SHA-256 KDF always produces 32 bytes; set length=32")
        return sha256(raw_secret)
    elif kdf == "hkdf-sha256":
        return hkdf_sha256(raw_secret, salt=salt, info=info, length=length)
    else:
        raise ValueError(f"Unsupported KDF: {kdf}. Use 'sha256' or 'hkdf-sha256'.")


def ecdh_verify_consistency(
    private_key_a: int,
    public_key_a: CurvePoint,
    private_key_b: int,
    public_key_b: CurvePoint,
    cofactor: bool = True,
    curve=Secp256r1,
) -> bool:
    validate_point(public_key_a, curve)
    validate_point(public_key_b, curve)

    secret_ab = ecdh_compute_shared_secret(private_key_a, public_key_b, cofactor, curve)
    secret_ba = ecdh_compute_shared_secret(private_key_b, public_key_a, cofactor, curve)

    return secret_ab == secret_ba
