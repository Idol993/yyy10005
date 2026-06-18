from typing import Tuple

from .curve import (
    CurvePoint,
    Secp256r1,
    INFINITY,
    is_point_on_curve,
    validate_point,
    scalar_mult,
)
from .field_ops import mod_mul, sha256


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
