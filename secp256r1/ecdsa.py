from typing import Optional, Tuple

from .field_ops import (
    mod_add,
    mod_sub,
    mod_mul,
    mod_inv,
    secure_random_int,
    constant_time_equal,
    constant_time_select,
    hmac_sha256,
    sha256,
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


def _int_to_bytes(value: int, length: int) -> bytes:
    return value.to_bytes(length, byteorder="big")


def _hash_message(message: bytes, n: int) -> int:
    hash_bytes = sha256(message)
    hash_int = int.from_bytes(hash_bytes, byteorder="big")
    n_bits = n.bit_length()
    hash_bits = len(hash_bytes) * 8
    if hash_bits > n_bits:
        hash_int >>= (hash_bits - n_bits)
    return hash_int


def _prehash_to_int(digest: bytes, n: int) -> int:
    if not isinstance(digest, (bytes, bytearray)):
        raise TypeError("Prehashed digest must be bytes")
    if len(digest) != 32:
        raise ValueError(f"Prehashed digest must be exactly 32 bytes (SHA-256), got {len(digest)}")
    hash_int = int.from_bytes(digest, byteorder="big")
    n_bits = n.bit_length()
    hash_bits = len(digest) * 8
    if hash_bits > n_bits:
        hash_int >>= (hash_bits - n_bits)
    return hash_int


def _bits2int(data: bytes, n: int) -> int:
    value = int.from_bytes(data, byteorder="big")
    data_bits = len(data) * 8
    n_bits = n.bit_length()
    if data_bits > n_bits:
        value >>= (data_bits - n_bits)
    return value


def _bits2octets(data: bytes, n: int) -> bytes:
    qlen = n.bit_length()
    rlen = (qlen + 7) // 8
    z = _bits2int(data, n)
    if z >= n:
        z = z - n
    return _int_to_bytes(z, rlen)


def _int2octets(value: int, n: int) -> bytes:
    rlen = (n.bit_length() + 7) // 8
    if value >= n:
        raise ValueError("int2octets: value >= n")
    return _int_to_bytes(value, rlen)


def _generate_k_rfc6979(
    private_key: int,
    message: bytes,
    n: int,
    hash_fn=sha256,
    hmac_fn=hmac_sha256,
) -> int:
    qlen = n.bit_length()
    rolen = (qlen + 7) // 8

    h1 = hash_fn(message)
    x = _int2octets(private_key, n)
    h1_prime = _bits2octets(h1, n)

    V = b"\x01" * 32
    K = b"\x00" * 32

    K = hmac_fn(K, V + b"\x00" + x + h1_prime)
    V = hmac_fn(K, V)
    K = hmac_fn(K, V + b"\x01" + x + h1_prime)
    V = hmac_fn(K, V)

    while True:
        T = b""
        while len(T) < rolen:
            V = hmac_fn(K, V)
            T = T + V

        k = _bits2int(T[:rolen], n)
        if 1 <= k < n:
            return k

        K = hmac_fn(K, V + b"\x00")
        V = hmac_fn(K, V)


def _generate_k_random(n: int) -> int:
    return secure_random_int(1, n - 1)


def sign(
    private_key: int,
    message: bytes,
    k: Optional[int] = None,
    deterministic: bool = False,
    prehashed: bool = False,
) -> Tuple[int, int]:
    if not isinstance(private_key, int):
        raise TypeError("Private key must be an integer")
    if private_key <= 0 or private_key >= Secp256r1.n:
        raise ValueError("Private key must be in range [1, n-1]")

    n = Secp256r1.n
    if prehashed:
        z = _prehash_to_int(message, n)
    else:
        if not isinstance(message, (bytes, bytearray)):
            raise TypeError("Message must be bytes")
        z = _hash_message(message, n)

    if k is not None and deterministic:
        raise ValueError("Cannot specify both k and deterministic=True")

    if k is not None:
        if not isinstance(k, int) or k <= 0 or k >= n:
            raise ValueError("k must be in range [1, n-1]")

    for attempt in range(100):
        if k is not None:
            k_val = k
        elif deterministic:
            if prehashed:
                raise ValueError("deterministic=True is not supported with prehashed=True")
            if attempt == 0:
                k_val = _generate_k_rfc6979(private_key, message, n)
            else:
                extra = b"\x00" * attempt
                k_val = _generate_k_rfc6979(private_key, message + extra, n)
        else:
            k_val = _generate_k_random(n)

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
    prehashed: bool = False,
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

    try:
        if prehashed:
            z = _prehash_to_int(message, n)
        else:
            if not isinstance(message, (bytes, bytearray)):
                raise TypeError("Message must be bytes")
            z = _hash_message(message, n)
    except (TypeError, ValueError):
        return False

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


def _validate_private_key(private_key: int) -> None:
    if not isinstance(private_key, int):
        raise TypeError("Private key must be an integer")
    if private_key <= 0 or private_key >= Secp256r1.n:
        raise ValueError("Private key out of valid range [1, n-1]")


def private_key_to_hex(private_key: int) -> str:
    _validate_private_key(private_key)
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
    _validate_private_key(private_key)
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
    if len(hex_str) == 0:
        raise ValueError("Empty public key hex string")
    if len(hex_str) % 2 != 0:
        raise ValueError("Public key hex string must have even length")
    try:
        key_bytes = bytes.fromhex(hex_str)
    except ValueError:
        raise ValueError("Invalid hex string for public key")

    byte_len = (Secp256r1.p.bit_length() + 7) // 8
    prefix = key_bytes[0]

    if prefix == 0x04:
        expected_len = 1 + 2 * byte_len
        if len(key_bytes) != expected_len:
            raise ValueError(
                f"Invalid uncompressed public key length: {len(key_bytes)}, expected {expected_len}"
            )
        x = int.from_bytes(key_bytes[1 : 1 + byte_len], byteorder="big")
        y = int.from_bytes(key_bytes[1 + byte_len :], byteorder="big")
    elif prefix in (0x02, 0x03):
        expected_len = 1 + byte_len
        if len(key_bytes) != expected_len:
            raise ValueError(
                f"Invalid compressed public key length: {len(key_bytes)}, expected {expected_len}"
            )
        x = int.from_bytes(key_bytes[1:], byteorder="big")
        y = _decompress_y(x, prefix == 0x03, Secp256r1)
    else:
        raise ValueError(f"Invalid public key prefix: 0x{prefix:02x}")

    if x < 0 or x >= Secp256r1.p or y < 0 or y >= Secp256r1.p:
        raise ValueError("Public key coordinates out of field range")

    point = CurvePoint(x, y, False)
    validate_point(point)
    return point


def _decompress_y(x: int, is_odd: bool, curve) -> int:
    if x < 0 or x >= curve.p:
        raise ValueError("x coordinate out of field range")

    p = curve.p
    a = curve.a
    b = curve.b

    rhs = (pow(x, 3, p) + a * x + b) % p
    y = pow(rhs, (p + 1) // 4, p)

    if pow(y, 2, p) != rhs:
        raise ValueError("Invalid compressed public key: x is not a valid x-coordinate on curve")

    if (y % 2 == 1) != is_odd:
        y = p - y

    return y


def _encode_der_integer_strict(val: int) -> bytes:
    if val < 0:
        raise ValueError("Negative integer not allowed in DER signature")
    if val == 0:
        return bytes([0x02, 0x01, 0x00])

    val_bytes = val.to_bytes((val.bit_length() + 7) // 8, byteorder="big")
    if val_bytes[0] & 0x80:
        val_bytes = b"\x00" + val_bytes
    else:
        if len(val_bytes) > 1 and val_bytes[0] == 0x00 and not (val_bytes[1] & 0x80):
            raise ValueError("Invalid DER integer: unnecessary leading 0x00 padding")
    length = len(val_bytes)
    if length > 0x7F:
        raise ValueError("Integer length exceeds single-byte DER length capacity")
    return bytes([0x02, length]) + val_bytes


def signature_to_der(signature: Tuple[int, int]) -> bytes:
    if not isinstance(signature, tuple) or len(signature) != 2:
        raise TypeError("Signature must be a tuple of (r, s)")
    r, s = signature
    if not isinstance(r, int) or not isinstance(s, int):
        raise TypeError("r and s must be integers")
    if r <= 0 or r >= Secp256r1.n or s <= 0 or s >= Secp256r1.n:
        raise ValueError("r and s must be in range [1, n-1]")

    r_bytes = _encode_der_integer_strict(r)
    s_bytes = _encode_der_integer_strict(s)
    content = r_bytes + s_bytes
    content_len = len(content)
    if content_len > 0x7F:
        raise ValueError("Signature content too long for short-form DER length")
    return bytes([0x30, content_len]) + content


def _parse_der_integer_strict(data: bytes, offset: int) -> Tuple[int, int]:
    if offset >= len(data):
        raise ValueError("Invalid DER: truncated at INTEGER tag")
    if data[offset] != 0x02:
        raise ValueError(f"Invalid DER: expected INTEGER tag 0x02, got 0x{data[offset]:02x}")

    if offset + 1 >= len(data):
        raise ValueError("Invalid DER: truncated at INTEGER length")
    int_len = data[offset + 1]

    if int_len & 0x80:
        raise ValueError("Invalid DER: INTEGER uses long-form length (not allowed for ECDSA sigs)")

    if int_len == 0:
        raise ValueError("Invalid DER: INTEGER with zero length")

    int_start = offset + 2
    if int_start + int_len > len(data):
        raise ValueError("Invalid DER: INTEGER value truncated")

    val_bytes = data[int_start : int_start + int_len]

    if val_bytes[0] & 0x80:
        raise ValueError("Invalid DER: negative INTEGER not allowed (missing 0x00 prefix for positive integer with high bit set)")

    if int_len > 1 and val_bytes[0] == 0x00 and not (val_bytes[1] & 0x80):
        raise ValueError("Invalid DER: unnecessary leading 0x00 padding on INTEGER")

    if int_len > 1 and val_bytes[0] == 0xFF and (val_bytes[1] & 0x80):
        raise ValueError("Invalid DER: negative INTEGER not allowed")

    value = int.from_bytes(val_bytes, byteorder="big")
    return value, int_start + int_len


def der_to_signature(der_bytes: bytes) -> Tuple[int, int]:
    if not isinstance(der_bytes, (bytes, bytearray)):
        raise TypeError("DER bytes must be bytes")
    der_bytes = bytes(der_bytes)

    if len(der_bytes) < 8:
        raise ValueError("DER data too short for a valid ECDSA signature")

    if der_bytes[0] != 0x30:
        raise ValueError(f"Invalid DER: expected SEQUENCE tag 0x30, got 0x{der_bytes[0]:02x}")

    seq_len_byte = der_bytes[1]
    if seq_len_byte & 0x80:
        raise ValueError("Invalid DER: SEQUENCE uses long-form length (not allowed)")

    expected_total = 2 + seq_len_byte
    if len(der_bytes) != expected_total:
        raise ValueError(
            f"Invalid DER: trailing garbage bytes (expected {expected_total} bytes, got {len(der_bytes)})"
        )

    seq_content = der_bytes[2:expected_total]

    r, offset_after_r = _parse_der_integer_strict(seq_content, 0)
    s, final_offset = _parse_der_integer_strict(seq_content, offset_after_r)

    if final_offset != len(seq_content):
        raise ValueError("Invalid DER: extra garbage after second INTEGER")

    if r <= 0 or r >= Secp256r1.n or s <= 0 or s >= Secp256r1.n:
        raise ValueError("Signature values out of valid range [1, n-1]")

    return (r, s)
