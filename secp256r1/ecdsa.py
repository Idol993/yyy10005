import hashlib
from typing import Optional, Tuple

from .field_ops import (
    mod_add,
    mod_sub,
    mod_mul,
    mod_inv,
    secure_random_int,
    constant_time_equal,
    constant_time_select,
)
from .curve import (
    CurvePoint,
    Secp256r1,
    INFINITY,
    is_point_on_curve,
    validate_point,
    scalar_mult,
    scalar_mult_base,
    point_add,
)


def generate_keypair() -> Tuple[int, CurvePoint]:
    private_key = secure_random_int(1, Secp256r1.n - 1)
    public_key = scalar_mult_base(private_key)
    validate_point(public_key)
    return private_key, public_key


def _hash_message(message: bytes, n: int) -> int:
    hash_bytes = hashlib.sha256(message).digest()
    hash_int = int.from_bytes(hash_bytes, byteorder="big")
    n_bits = n.bit_length()
    hash_bits = len(hash_bytes) * 8
    if hash_bits > n_bits:
        hash_int >>= (hash_bits - n_bits)
    return hash_int


def _generate_k_random(n: int) -> int:
    return secure_random_int(1, n - 1)


def sign(
    private_key: int,
    message: bytes,
    k: Optional[int] = None,
) -> Tuple[int, int]:
    if not isinstance(private_key, int):
        raise TypeError("Private key must be an integer")
    if private_key <= 0 or private_key >= Secp256r1.n:
        raise ValueError("Private key must be in range [1, n-1]")
    if not isinstance(message, (bytes, bytearray)):
        raise TypeError("Message must be bytes")

    n = Secp256r1.n
    z = _hash_message(message, n)

    for _ in range(100):
        if k is None:
            k_val = _generate_k_random(n)
        else:
            if not isinstance(k, int) or k <= 0 or k >= n:
                raise ValueError("k must be in range [1, n-1]")
            k_val = k

        R = scalar_mult_base(k_val)

        if R.infinity:
            if k is not None:
                raise ValueError("k resulted in point at infinity")
            continue

        r = R.x % n
        if r == 0:
            if k is not None:
                raise ValueError("r is zero")
            continue

        k_inv = mod_inv(k_val, n)
        s = mod_mul(k_inv, mod_add(z, mod_mul(private_key, r, n), n), n)

        if s == 0:
            if k is not None:
                raise ValueError("s is zero")
            continue

        if s > n // 2:
            s = n - s

        return (r, s)

    raise RuntimeError("Failed to generate valid signature after 100 attempts")


def verify(
    public_key: CurvePoint,
    message: bytes,
    signature: Tuple[int, int],
) -> bool:
    try:
        validate_point(public_key)
    except (TypeError, ValueError):
        return False

    if not isinstance(signature, tuple) or len(signature) != 2:
        return False

    r, s = signature

    if not isinstance(r, int) or not isinstance(s, int):
        return False

    n = Secp256r1.n
    if r < 1 or r >= n or s < 1 or s >= n:
        return False

    if not isinstance(message, (bytes, bytearray)):
        raise TypeError("Message must be bytes")

    z = _hash_message(message, n)

    try:
        s_inv = mod_inv(s, n)
    except ValueError:
        return False

    u1 = mod_mul(z, s_inv, n)
    u2 = mod_mul(r, s_inv, n)

    point1 = scalar_mult_base(u1)
    point2 = scalar_mult(u2, public_key)

    P = point_add(point1, point2)

    if P.infinity:
        return False

    v = P.x % n

    result = constant_time_equal(v, r)
    return bool(result)


def private_key_to_hex(private_key: int) -> str:
    if not isinstance(private_key, int):
        raise TypeError("Private key must be an integer")
    if private_key < 0 or private_key >= Secp256r1.n:
        raise ValueError("Private key out of valid range")
    byte_len = (Secp256r1.n.bit_length() + 7) // 8
    return private_key.to_bytes(byte_len, byteorder="big").hex()


def hex_to_private_key(hex_str: str) -> int:
    if not isinstance(hex_str, str):
        raise TypeError("Hex string must be a string")
    hex_str = hex_str.strip()
    if hex_str.startswith("0x") or hex_str.startswith("0X"):
        hex_str = hex_str[2:]
    if len(hex_str) != 64:
        raise ValueError(f"Invalid private key hex length: {len(hex_str)}, expected 64")
    try:
        private_key = int(hex_str, 16)
    except ValueError:
        raise ValueError("Invalid hex string for private key")
    if private_key <= 0 or private_key >= Secp256r1.n:
        raise ValueError("Private key out of valid range")
    return private_key


def public_key_to_hex(public_key: CurvePoint, compressed: bool = False) -> str:
    validate_point(public_key)
    byte_len = (Secp256r1.p.bit_length() + 7) // 8
    if compressed:
        prefix = b"\x02" if (public_key.y % 2 == 0) else b"\x03"
        return (prefix + public_key.x.to_bytes(byte_len, byteorder="big")).hex()
    else:
        prefix = b"\x04"
        return (
            prefix
            + public_key.x.to_bytes(byte_len, byteorder="big")
            + public_key.y.to_bytes(byte_len, byteorder="big")
        ).hex()


def hex_to_public_key(hex_str: str) -> CurvePoint:
    if not isinstance(hex_str, str):
        raise TypeError("Hex string must be a string")
    hex_str = hex_str.strip()
    if hex_str.startswith("0x") or hex_str.startswith("0X"):
        hex_str = hex_str[2:]
    try:
        key_bytes = bytes.fromhex(hex_str)
    except ValueError:
        raise ValueError("Invalid hex string for public key")

    if len(key_bytes) == 0:
        raise ValueError("Empty public key data")

    byte_len = (Secp256r1.p.bit_length() + 7) // 8
    prefix = key_bytes[0]

    if prefix == 0x04:
        if len(key_bytes) != 1 + 2 * byte_len:
            raise ValueError(
                f"Invalid uncompressed public key length: {len(key_bytes)}"
            )
        x = int.from_bytes(key_bytes[1 : 1 + byte_len], byteorder="big")
        y = int.from_bytes(key_bytes[1 + byte_len :], byteorder="big")
    elif prefix in (0x02, 0x03):
        if len(key_bytes) != 1 + byte_len:
            raise ValueError(
                f"Invalid compressed public key length: {len(key_bytes)}"
            )
        x = int.from_bytes(key_bytes[1:], byteorder="big")
        y = _decompress_y(x, prefix == 0x03, Secp256r1)
    else:
        raise ValueError(f"Invalid public key prefix: 0x{prefix:02x}")

    point = CurvePoint(x, y, False)
    validate_point(point)
    return point


def _decompress_y(x: int, is_odd: bool, curve) -> int:
    p = curve.p
    a = curve.a
    b = curve.b

    rhs = (pow(x, 3, p) + a * x + b) % p
    y = pow(rhs, (p + 1) // 4, p)

    if pow(y, 2, p) != rhs:
        raise ValueError("Invalid compressed public key: cannot recover y")

    if (y % 2 == 1) != is_odd:
        y = p - y

    return y


def signature_to_der(signature: Tuple[int, int]) -> bytes:
    if not isinstance(signature, tuple) or len(signature) != 2:
        raise TypeError("Signature must be a tuple of (r, s)")
    r, s = signature

    def _encode_integer(val: int) -> bytes:
        val_bytes = val.to_bytes((val.bit_length() + 7) // 8 or 1, byteorder="big")
        if val_bytes[0] & 0x80:
            val_bytes = b"\x00" + val_bytes
        return bytes([0x02, len(val_bytes)]) + val_bytes

    r_bytes = _encode_integer(r)
    s_bytes = _encode_integer(s)
    content = r_bytes + s_bytes
    return bytes([0x30, len(content)]) + content


def der_to_signature(der_bytes: bytes) -> Tuple[int, int]:
    if not isinstance(der_bytes, (bytes, bytearray)):
        raise TypeError("DER bytes must be bytes")
    der_bytes = bytes(der_bytes)

    if len(der_bytes) < 8:
        raise ValueError("DER data too short")
    if der_bytes[0] != 0x30:
        raise ValueError("Invalid DER: missing SEQUENCE tag")

    seq_len = der_bytes[1]
    seq_start = 2
    if seq_len & 0x80:
        len_bytes = seq_len & 0x7F
        if len(der_bytes) < 2 + len_bytes:
            raise ValueError("Invalid DER: malformed length")
        seq_len = int.from_bytes(der_bytes[2 : 2 + len_bytes], byteorder="big")
        seq_start = 2 + len_bytes

    if len(der_bytes) < seq_start + seq_len:
        raise ValueError("Invalid DER: sequence length exceeds data")

    def _parse_integer(data: bytes, offset: int) -> Tuple[int, int]:
        if offset >= len(data) or data[offset] != 0x02:
            raise ValueError("Invalid DER: missing INTEGER tag")
        int_len = data[offset + 1]
        int_start = offset + 2
        if int_len & 0x80:
            len_bytes_count = int_len & 0x7F
            if len(data) < int_start + len_bytes_count:
                raise ValueError("Invalid DER: malformed integer length")
            int_len = int.from_bytes(
                data[int_start : int_start + len_bytes_count], byteorder="big"
            )
            int_start = int_start + len_bytes_count
        if len(data) < int_start + int_len:
            raise ValueError("Invalid DER: integer length exceeds data")
        val = int.from_bytes(data[int_start : int_start + int_len], byteorder="big")
        return val, int_start + int_len

    r, offset_after_r = _parse_integer(der_bytes, seq_start)
    s, _ = _parse_integer(der_bytes, offset_after_r)

    if r < 0 or r >= Secp256r1.n or s < 0 or s >= Secp256r1.n:
        raise ValueError("Signature values out of range")

    return (r, s)
