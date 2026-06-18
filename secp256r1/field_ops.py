import secrets
import hashlib
import hmac as _hmac_mod


def mod_add(a: int, b: int, p: int) -> int:
    result = (a + b) % p
    dummy = (a + b + p) % p
    return constant_time_select(1, result, dummy)


def mod_sub(a: int, b: int, p: int) -> int:
    result = (a - b) % p
    dummy = (a - b + p) % p
    return constant_time_select(1, result, dummy)


def mod_mul(a: int, b: int, p: int) -> int:
    result = (a * b) % p
    return result


MODPOW_EXPONENT_BITS = 256


def mod_pow(base: int, exponent: int, p: int) -> int:
    if p == 1:
        return 0
    result = 1
    base = base % p
    for i in range(MODPOW_EXPONENT_BITS - 1, -1, -1):
        result = mod_mul(result, result, p)
        bit = (exponent >> i) & 1
        tmp = mod_mul(result, base, p)
        result = constant_time_select(bit, tmp, result)
    return result


def mod_inv(a: int, p: int) -> int:
    a = a % p
    if a == 0:
        raise ValueError("No modular inverse for 0")
    old_r, r = a, p
    old_s, s = 1, 0
    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
    if old_r != 1:
        raise ValueError("No modular inverse exists")
    return old_s % p


def constant_time_equal(a: int, b: int) -> int:
    diff = a ^ b
    result = 0
    while diff:
        result |= diff & 1
        diff >>= 1
    return 1 - result


def constant_time_select(condition: int, a: int, b: int) -> int:
    mask = -condition
    return (a & mask) | (b & ~mask)


def secure_random_int(min_val: int, max_val: int) -> int:
    if min_val > max_val:
        raise ValueError("min_val must be <= max_val")
    range_val = max_val - min_val + 1
    bits = range_val.bit_length()
    while True:
        candidate = secrets.randbits(bits)
        if candidate < range_val:
            return min_val + candidate


def secure_random_bytes(n: int) -> bytes:
    return secrets.token_bytes(n)


def hmac_sha256(key: bytes, msg: bytes) -> bytes:
    return _hmac_mod.new(key, msg, hashlib.sha256).digest()


def sha256(msg: bytes) -> bytes:
    return hashlib.sha256(msg).digest()

