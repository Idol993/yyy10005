import base64
from typing import Tuple

from .curve import CurvePoint, Secp256r1, INFINITY, is_point_on_curve, validate_point, scalar_mult_base
from .ecdsa import (
    private_key_to_hex,
    hex_to_private_key,
    public_key_to_hex,
    hex_to_public_key,
    _validate_private_key,
)


OID_EC_PUBLIC_KEY = bytes([0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x02, 0x01])
OID_PRIME256V1 = bytes([0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x03, 0x01, 0x07])
OID_ECDH = bytes([0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x01, 0x0C])

PEM_LABEL_EC_PRIVATE_KEY = "EC PRIVATE KEY"
PEM_LABEL_PRIVATE_KEY = "PRIVATE KEY"
PEM_LABEL_PUBLIC_KEY = "PUBLIC KEY"


def _encode_der_length(length: int) -> bytes:
    if length < 0x80:
        return bytes([length])
    elif length <= 0xFF:
        return bytes([0x81, length])
    elif length <= 0xFFFF:
        return bytes([0x82, (length >> 8) & 0xFF, length & 0xFF])
    else:
        raise ValueError("DER length too large")


def _decode_der_length(data: bytes, offset: int) -> Tuple[int, int]:
    if offset >= len(data):
        raise ValueError("Truncated DER length")
    first = data[offset]
    if first & 0x80 == 0:
        return first, offset + 1
    num_bytes = first & 0x7F
    if num_bytes == 0:
        raise ValueError("Indefinite DER length not supported")
    if offset + 1 + num_bytes > len(data):
        raise ValueError("Truncated DER long-form length")
    length = 0
    for i in range(num_bytes):
        length = (length << 8) | data[offset + 1 + i]
    return length, offset + 1 + num_bytes


def _der_sequence(elements: list) -> bytes:
    content = b"".join(elements)
    return b"\x30" + _encode_der_length(len(content)) + content


def _der_octet_string(data: bytes) -> bytes:
    return b"\x04" + _encode_der_length(len(data)) + data


def _der_integer(value: int) -> bytes:
    if value < 0:
        raise ValueError("Negative integers not supported")
    if value == 0:
        return b"\x02\x01\x00"
    val_bytes = value.to_bytes((value.bit_length() + 7) // 8, byteorder="big")
    if val_bytes[0] & 0x80:
        val_bytes = b"\x00" + val_bytes
    return b"\x02" + _encode_der_length(len(val_bytes)) + val_bytes


def _der_bit_string(data: bytes, unused_bits: int = 0) -> bytes:
    content = bytes([unused_bits]) + data
    return b"\x03" + _encode_der_length(len(content)) + content


def _der_oid(oid_bytes: bytes) -> bytes:
    return b"\x06" + _encode_der_length(len(oid_bytes)) + oid_bytes


def _der_tag(tag: int, content: bytes, constructed: bool = False) -> bytes:
    tag_byte = tag | (0x20 if constructed else 0x00)
    return bytes([tag_byte]) + _encode_der_length(len(content)) + content


def _parse_der_integer(data: bytes, offset: int) -> Tuple[int, int]:
    if offset >= len(data) or data[offset] != 0x02:
        raise ValueError(f"Expected INTEGER tag at offset {offset}, got 0x{data[offset]:02x}")
    length, next_offset = _decode_der_length(data, offset + 1)
    if next_offset + length > len(data):
        raise ValueError("Truncated INTEGER value")
    val_bytes = data[next_offset : next_offset + length]
    if length == 0:
        raise ValueError("INTEGER with zero length")
    if length > 1 and val_bytes[0] == 0 and not (val_bytes[1] & 0x80):
        raise ValueError("Unnecessary leading 0x00 padding on INTEGER")
    if val_bytes[0] & 0x80:
        raise ValueError("Negative INTEGER not allowed")
    value = int.from_bytes(val_bytes, byteorder="big")
    return value, next_offset + length


def _parse_der_octet_string(data: bytes, offset: int) -> Tuple[bytes, int]:
    if offset >= len(data) or data[offset] != 0x04:
        raise ValueError(f"Expected OCTET STRING tag at offset {offset}, got 0x{data[offset]:02x}")
    length, next_offset = _decode_der_length(data, offset + 1)
    if next_offset + length > len(data):
        raise ValueError("Truncated OCTET STRING value")
    return data[next_offset : next_offset + length], next_offset + length


def _parse_der_sequence(data: bytes, offset: int) -> Tuple[int, int, int]:
    if offset >= len(data) or data[offset] != 0x30:
        raise ValueError(f"Expected SEQUENCE tag at offset {offset}, got 0x{data[offset]:02x}")
    length, next_offset = _decode_der_length(data, offset + 1)
    if next_offset + length > len(data):
        raise ValueError("Truncated SEQUENCE value")
    return next_offset, length, next_offset + length


def _parse_der_oid(data: bytes, offset: int) -> Tuple[bytes, int]:
    if offset >= len(data) or data[offset] != 0x06:
        raise ValueError(f"Expected OID tag at offset {offset}, got 0x{data[offset]:02x}")
    length, next_offset = _decode_der_length(data, offset + 1)
    if next_offset + length > len(data):
        raise ValueError("Truncated OID value")
    return data[next_offset : next_offset + length], next_offset + length


def _parse_der_bit_string(data: bytes, offset: int) -> Tuple[bytes, int, int]:
    if offset >= len(data) or data[offset] != 0x03:
        raise ValueError(f"Expected BIT STRING tag at offset {offset}, got 0x{data[offset]:02x}")
    length, next_offset = _decode_der_length(data, offset + 1)
    if next_offset + length > len(data):
        raise ValueError("Truncated BIT STRING value")
    if length < 1:
        raise ValueError("BIT STRING too short")
    unused_bits = data[next_offset]
    if unused_bits != 0:
        raise ValueError("Non-zero unused bits in BIT STRING not supported")
    content = data[next_offset + 1 : next_offset + length]
    return content, unused_bits, next_offset + length


def _parse_der_tag(data: bytes, offset: int) -> Tuple[int, bytes, int]:
    if offset >= len(data):
        raise ValueError("Truncated tag")
    tag = data[offset]
    length, next_offset = _decode_der_length(data, offset + 1)
    if next_offset + length > len(data):
        raise ValueError("Truncated tag content")
    content = data[next_offset : next_offset + length]
    return tag, content, next_offset + length


def _encode_sec1_private_key(private_key: int, public_key: CurvePoint = None) -> bytes:
    _validate_private_key(private_key)
    byte_len = 32

    priv_bytes = private_key.to_bytes(byte_len, byteorder="big")

    version = _der_integer(1)
    priv_octet = _der_octet_string(priv_bytes)

    optional_parts = []
    params = _der_sequence([_der_oid(OID_EC_PUBLIC_KEY), _der_oid(OID_PRIME256V1)])
    optional_parts.append(_der_tag(0xA0, params, constructed=True))

    if public_key is not None:
        validate_point(public_key)
        pub_hex = public_key_to_hex(public_key, compressed=False)
        pub_bytes = bytes.fromhex(pub_hex)
        pub_bit = _der_bit_string(pub_bytes)
        optional_parts.append(_der_tag(0xA1, pub_bit, constructed=True))

    return _der_sequence([version, priv_octet] + optional_parts)


def _decode_sec1_private_key(der_bytes: bytes) -> Tuple[int, CurvePoint]:
    seq_start, seq_len, seq_end = _parse_der_sequence(der_bytes, 0)
    if seq_end != len(der_bytes):
        raise ValueError("Trailing garbage after SEC1 private key SEQUENCE")

    offset = seq_start

    version, offset = _parse_der_integer(der_bytes, offset)
    if version != 1:
        raise ValueError(f"Unsupported SEC1 private key version: {version}")

    priv_bytes, offset = _parse_der_octet_string(der_bytes, offset)
    if len(priv_bytes) != 32:
        raise ValueError(f"Invalid SEC1 private key length: {len(priv_bytes)}, expected 32")

    private_key = int.from_bytes(priv_bytes, byteorder="big")
    _validate_private_key(private_key)

    public_key = None
    params_found = False

    while offset < seq_start + seq_len:
        tag, content, next_offset = _parse_der_tag(der_bytes, offset)

        if tag == 0xA0:
            inner_start, inner_len, inner_end = _parse_der_sequence(content, 0)
            if inner_end != len(content):
                raise ValueError("Trailing garbage in EC parameters")
            oid1, oid_offset = _parse_der_oid(content, inner_start)
            if oid1 != OID_EC_PUBLIC_KEY:
                raise ValueError(f"Unexpected algorithm OID: {oid1.hex()}, expected ecPublicKey")
            oid2, _ = _parse_der_oid(content, oid_offset)
            if oid2 != OID_PRIME256V1:
                raise ValueError(f"Unsupported curve OID: {oid2.hex()}, expected prime256v1 (secp256r1)")
            params_found = True
        elif tag == 0xA1:
            pub_raw, _, _ = _parse_der_bit_string(content, 0)
            public_key = hex_to_public_key(pub_raw.hex())
        else:
            raise ValueError(f"Unexpected tag in SEC1 private key: 0x{tag:02x}")

        offset = next_offset

    if not params_found:
        raise ValueError("SEC1 private key missing EC parameters (curve OID)")

    if public_key is None:
        public_key = scalar_mult_base(private_key)

    return private_key, public_key


def _encode_pkcs8_private_key(private_key: int, public_key: CurvePoint = None) -> bytes:
    _validate_private_key(private_key)

    version = _der_integer(0)

    algorithm_oid = _der_oid(OID_EC_PUBLIC_KEY)
    curve_oid = _der_oid(OID_PRIME256V1)
    algorithm_id = _der_sequence([algorithm_oid, curve_oid])

    sec1_der = _encode_sec1_private_key(private_key, public_key)
    priv_octet = _der_octet_string(sec1_der)

    return _der_sequence([version, algorithm_id, priv_octet])


def _decode_pkcs8_private_key(der_bytes: bytes) -> Tuple[int, CurvePoint]:
    seq_start, seq_len, seq_end = _parse_der_sequence(der_bytes, 0)
    if seq_end != len(der_bytes):
        raise ValueError("Trailing garbage after PKCS#8 SEQUENCE")

    offset = seq_start
    version, offset = _parse_der_integer(der_bytes, offset)
    if version != 0:
        raise ValueError(f"Unsupported PKCS#8 version: {version}")

    algo_start, algo_len, algo_end = _parse_der_sequence(der_bytes, offset)
    offset = algo_end

    oid1, oid_offset = _parse_der_oid(der_bytes, algo_start)
    if oid1 != OID_EC_PUBLIC_KEY:
        raise ValueError(f"Unexpected algorithm OID: {oid1.hex()}, expected ecPublicKey")
    oid2, _ = _parse_der_oid(der_bytes, oid_offset)
    if oid2 != OID_PRIME256V1:
        raise ValueError(f"Unsupported curve OID: {oid2.hex()}, expected prime256v1 (secp256r1)")

    priv_bytes, offset = _parse_der_octet_string(der_bytes, offset)
    return _decode_sec1_private_key(priv_bytes)


def _encode_spki_public_key(public_key: CurvePoint) -> bytes:
    validate_point(public_key)

    algorithm_oid = _der_oid(OID_EC_PUBLIC_KEY)
    curve_oid = _der_oid(OID_PRIME256V1)
    algorithm_id = _der_sequence([algorithm_oid, curve_oid])

    pub_hex = public_key_to_hex(public_key, compressed=False)
    pub_bytes = bytes.fromhex(pub_hex)
    pub_bit = _der_bit_string(pub_bytes)

    return _der_sequence([algorithm_id, pub_bit])


def _decode_spki_public_key(der_bytes: bytes) -> CurvePoint:
    seq_start, seq_len, seq_end = _parse_der_sequence(der_bytes, 0)
    if seq_end != len(der_bytes):
        raise ValueError("Trailing garbage after SPKI SEQUENCE")

    offset = seq_start
    algo_start, algo_len, algo_end = _parse_der_sequence(der_bytes, offset)
    offset = algo_end

    oid1, oid_offset = _parse_der_oid(der_bytes, algo_start)
    if oid1 != OID_EC_PUBLIC_KEY:
        raise ValueError(f"Unexpected algorithm OID: {oid1.hex()}, expected ecPublicKey")
    oid2, _ = _parse_der_oid(der_bytes, oid_offset)
    if oid2 != OID_PRIME256V1:
        raise ValueError(f"Unsupported curve OID: {oid2.hex()}, expected prime256v1 (secp256r1)")

    pub_raw, _, _ = _parse_der_bit_string(der_bytes, offset)
    return hex_to_public_key(pub_raw.hex())


def _pem_encode(label: str, der_bytes: bytes) -> str:
    b64 = base64.b64encode(der_bytes).decode("ascii")
    lines = [b64[i : i + 64] for i in range(0, len(b64), 64)]
    body = "\n".join(lines)
    return f"-----BEGIN {label}-----\n{body}\n-----END {label}-----\n"


def _pem_decode(label: str, pem_text: str) -> bytes:
    if not isinstance(pem_text, (str, bytes)):
        raise TypeError("PEM text must be a string or bytes")
    if isinstance(pem_text, bytes):
        pem_text = pem_text.decode("ascii", errors="strict")

    begin_marker = f"-----BEGIN {label}-----"
    end_marker = f"-----END {label}-----"

    pem_text = pem_text.strip()

    if not pem_text.startswith(begin_marker):
        raise ValueError(f"PEM must start with BEGIN {label} marker (no extra text before it)")

    end_idx = pem_text.find(end_marker, len(begin_marker))
    if end_idx == -1:
        raise ValueError(f"PEM missing END {label} marker")

    after_end = end_idx + len(end_marker)
    remaining = pem_text[after_end:].strip()
    if remaining:
        raise ValueError(f"PEM has extra text after END {label} marker")

    b64_content = pem_text[len(begin_marker) : end_idx]
    b64_clean = "".join(b64_content.split())

    if not b64_clean:
        raise ValueError("PEM contains no data between markers")

    if not all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in b64_clean):
        raise ValueError("PEM contains invalid base64 characters")

    try:
        return base64.b64decode(b64_clean)
    except Exception:
        raise ValueError("Invalid base64 encoding in PEM")


def private_key_to_pem(
    private_key: int,
    public_key: CurvePoint = None,
    format: str = "pkcs8",
) -> str:
    _validate_private_key(private_key)

    if public_key is not None:
        validate_point(public_key)
        expected_pub = scalar_mult_base(private_key)
        if expected_pub != public_key:
            raise ValueError("Provided public key does not match private key")

    if format == "sec1":
        der = _encode_sec1_private_key(private_key, public_key)
        return _pem_encode(PEM_LABEL_EC_PRIVATE_KEY, der)
    elif format == "pkcs8":
        der = _encode_pkcs8_private_key(private_key, public_key)
        return _pem_encode(PEM_LABEL_PRIVATE_KEY, der)
    else:
        raise ValueError(f"Unsupported private key format: {format}. Use 'sec1' or 'pkcs8'.")


def pem_to_private_key(pem_text: str) -> Tuple[int, CurvePoint, str]:
    for label, decoder in [
        (PEM_LABEL_PRIVATE_KEY, _decode_pkcs8_private_key),
        (PEM_LABEL_EC_PRIVATE_KEY, _decode_sec1_private_key),
    ]:
        try:
            der = _pem_decode(label, pem_text)
            priv, pub = decoder(der)
            return priv, pub, "pkcs8" if label == PEM_LABEL_PRIVATE_KEY else "sec1"
        except ValueError:
            continue

    raise ValueError(
        "Could not decode PEM as PKCS#8 (PRIVATE KEY) or SEC1 (EC PRIVATE KEY). "
        "Check that the PEM is for an secp256r1 (prime256v1) key."
    )


def public_key_to_pem(public_key: CurvePoint) -> str:
    validate_point(public_key)
    der = _encode_spki_public_key(public_key)
    return _pem_encode(PEM_LABEL_PUBLIC_KEY, der)


def pem_to_public_key(pem_text: str) -> CurvePoint:
    der = _pem_decode(PEM_LABEL_PUBLIC_KEY, pem_text)
    return _decode_spki_public_key(der)
