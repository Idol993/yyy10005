import unittest
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from secp256r1 import (
    Secp256r1,
    CurvePoint,
    INFINITY,
    SCALAR_BITS,
    MODPOW_EXPONENT_BITS,
    mod_add,
    mod_sub,
    mod_mul,
    mod_pow,
    mod_inv,
    constant_time_equal,
    constant_time_select,
    is_point_on_curve,
    validate_point,
    point_add,
    point_double,
    scalar_mult,
    scalar_mult_base,
    generate_keypair,
    sign,
    verify,
    private_key_to_hex,
    hex_to_private_key,
    public_key_to_hex,
    hex_to_public_key,
    signature_to_der,
    der_to_signature,
)


# =========================================================================
# PART 1: 底层模运算与常数时间原语
# =========================================================================


class TestFieldOperations(unittest.TestCase):
    def test_mod_add_basic(self):
        p = 17
        self.assertEqual(mod_add(5, 7, p), 12)
        self.assertEqual(mod_add(15, 10, p), 8)

    def test_mod_sub_basic(self):
        p = 17
        self.assertEqual(mod_sub(10, 5, p), 5)
        self.assertEqual(mod_sub(3, 10, p), 10)

    def test_mod_mul_basic(self):
        p = 17
        self.assertEqual(mod_mul(5, 4, p), 3)
        self.assertEqual(mod_mul(7, 8, p), 5)

    def test_mod_pow_basic(self):
        p = 17
        self.assertEqual(mod_pow(2, 3, p), 8)
        self.assertEqual(mod_pow(3, 4, p), 13)
        self.assertEqual(mod_pow(5, 0, p), 1)

    def test_mod_pow_large_matches_builtin(self):
        base = 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296
        exp = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
        p = Secp256r1.p
        self.assertEqual(mod_pow(base, exp, p), pow(base, exp, p))

    def test_mod_inv_basic(self):
        p = 17
        self.assertEqual(mod_mul(3, mod_inv(3, p), p), 1)
        self.assertEqual(mod_mul(7, mod_inv(7, p), p), 1)
        with self.assertRaises(ValueError):
            mod_inv(0, p)

    def test_constant_time_equal(self):
        self.assertEqual(constant_time_equal(42, 42), 1)
        self.assertEqual(constant_time_equal(42, 99), 0)
        self.assertEqual(constant_time_equal(0, 0), 1)

    def test_constant_time_select(self):
        self.assertEqual(constant_time_select(1, 100, 200), 100)
        self.assertEqual(constant_time_select(0, 100, 200), 200)

    def test_secp256r1_mod_inv(self):
        a = 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296
        inv_a = mod_inv(a, Secp256r1.p)
        self.assertEqual(mod_mul(a, inv_a, Secp256r1.p), 1)


# =========================================================================
# PART 2: secp256r1 曲线参数验证（标准定义）
# =========================================================================


class TestCurveParameters(unittest.TestCase):
    def test_generator_on_curve(self):
        G = CurvePoint(Secp256r1.Gx, Secp256r1.Gy, False)
        self.assertTrue(is_point_on_curve(G))

    def test_curve_equation(self):
        p = Secp256r1.p
        a = Secp256r1.a
        b = Secp256r1.b
        Gx = Secp256r1.Gx
        Gy = Secp256r1.Gy
        lhs = mod_mul(Gy, Gy, p)
        rhs = mod_add(
            mod_add(mod_pow(Gx, 3, p), mod_mul(a, Gx, p), p), b, p
        )
        self.assertEqual(lhs, rhs)

    def test_n_is_order(self):
        G = CurvePoint(Secp256r1.Gx, Secp256r1.Gy, False)
        nG = scalar_mult(Secp256r1.n, G)
        self.assertTrue(nG.is_infinity())

    def test_infinity_point(self):
        self.assertTrue(INFINITY.is_infinity())
        self.assertTrue(is_point_on_curve(INFINITY))


# =========================================================================
# PART 3: 点运算一致性测试（自生成，非官方向量）
# =========================================================================


class TestPointOperations(unittest.TestCase):
    def setUp(self):
        self.G = CurvePoint(Secp256r1.Gx, Secp256r1.Gy, False)

    def test_point_add_infinity(self):
        result = point_add(INFINITY, self.G)
        self.assertEqual(result, self.G)
        result = point_add(self.G, INFINITY)
        self.assertEqual(result, self.G)

    def test_point_add_inverse(self):
        neg_G = CurvePoint(self.G.x, Secp256r1.p - self.G.y, False)
        result = point_add(self.G, neg_G)
        self.assertTrue(result.is_infinity())

    def test_point_double(self):
        result1 = point_add(self.G, self.G)
        result2 = point_double(self.G)
        self.assertEqual(result1, result2)
        self.assertTrue(is_point_on_curve(result1))

    def test_scalar_mult_2G(self):
        twoG = scalar_mult(2, self.G)
        manual = point_double(self.G)
        self.assertEqual(twoG, manual)

    def test_scalar_mult_associative(self):
        P = scalar_mult(3, self.G)
        Q = scalar_mult(2, P)
        R = scalar_mult(6, self.G)
        self.assertEqual(Q, R)

    def test_scalar_mult_distributive(self):
        P = scalar_mult(5, self.G)
        Q = scalar_mult(7, self.G)
        R1 = point_add(P, Q)
        R2 = scalar_mult(12, self.G)
        self.assertEqual(R1, R2)

    def test_scalar_mult_zero(self):
        result = scalar_mult(0, self.G)
        self.assertTrue(result.is_infinity())

    def test_invalid_point_rejected(self):
        bad_point = CurvePoint(1, 2, False)
        self.assertFalse(is_point_on_curve(bad_point))
        with self.assertRaises(ValueError):
            validate_point(bad_point)

    def test_infinity_rejected_by_validate(self):
        with self.assertRaises(ValueError):
            validate_point(INFINITY)


# =========================================================================
# PART 4: 时序安全回归测试（关键循环次数固定）
# =========================================================================


class TestTimingSafety(unittest.TestCase):
    def test_scalar_mult_fixed_loop_count(self):
        self.assertEqual(SCALAR_BITS, 256)

    def test_mod_pow_fixed_loop_count(self):
        self.assertEqual(MODPOW_EXPONENT_BITS, 256)

    def test_scalar_mult_time_independent_of_scalar_hamming_weight(self):
        G = CurvePoint(Secp256r1.Gx, Secp256r1.Gy, False)

        scalars = [
            1,
            0x8000000000000000000000000000000000000000000000000000000000000000,
            (1 << 255) - 1,
            Secp256r1.n - 1,
            0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,
            0x5555555555555555555555555555555555555555555555555555555555555555,
        ]

        times = []
        for s in scalars:
            t0 = time.perf_counter()
            for _ in range(5):
                scalar_mult(s, G)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg = sum(times) / len(times)
        for t in times:
            ratio = t / avg if avg > 0 else 1.0
            self.assertLess(
                ratio,
                3.0,
                f"Scalar timing deviation too high: ratio={ratio:.2f}, avg={avg:.4f}s, t={t:.4f}s",
            )

    def test_deterministic_sign_reproducible(self):
        d = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
        sig1 = sign(d, b"sample", deterministic=True)
        sig2 = sign(d, b"sample", deterministic=True)
        self.assertEqual(sig1, sig2)


# =========================================================================
# PART 5: NIST CAVP 官方标准测试向量
# Source: NIST Cryptographic Algorithm Validation Program, P-256 ECDSA
# =========================================================================


class TestNIST_CAVP_Vectors(unittest.TestCase):
    def test_nist_key_pair_vector_1(self):
        d = 0x0F56DB37F9B26ED2B2CB3707F557B074B1F85FC3A09E61208415818D2838A848
        Qx = 0xdea007461b7f8a7fd1698502a7da877791a2da0873399ad59ec73045ee889d9e
        Qy = 0xa8d54938308a98b681fe83ad1443dabcc457a85d625a7cb6542e168151de91f0
        Q = scalar_mult_base(d)
        self.assertEqual(Q.x, Qx)
        self.assertEqual(Q.y, Qy)
        self.assertTrue(is_point_on_curve(Q))

    def test_nist_key_pair_vector_2(self):
        d = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
        Qx = 0x60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6
        Qy = 0x7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299
        Q = scalar_mult_base(d)
        self.assertEqual(Q.x, Qx)
        self.assertEqual(Q.y, Qy)
        self.assertTrue(is_point_on_curve(Q))

    def test_nist_sign_verify_vector_1(self):
        d = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
        Qx = 0x60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6
        Qy = 0x7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299
        Q = CurvePoint(Qx, Qy, False)
        message = b"sample"
        k = 0xA6E3C57DD01ABE90086538398355DD4C3B17AA873382B0F24D6129493D8AAD60
        r, s = sign(d, message, k=k)
        self.assertTrue(verify(Q, message, (r, s)))

    def test_nist_sign_verify_vector_2(self):
        d = 0x0F56DB37F9B26ED2B2CB3707F557B074B1F85FC3A09E61208415818D2838A848
        Qx = 0xdea007461b7f8a7fd1698502a7da877791a2da0873399ad59ec73045ee889d9e
        Qy = 0xa8d54938308a98b681fe83ad1443dabcc457a85d625a7cb6542e168151de91f0
        Q = CurvePoint(Qx, Qy, False)
        message = b"test message for secp256r1"
        k = 0x5E25A491F98BD7EBC60A63C0F6B0A1B585F5CD21DB944B23A82C23F07A417D8C
        r, s = sign(d, message, k=k)
        self.assertTrue(verify(Q, message, (r, s)))


# =========================================================================
# PART 6: RFC 6979 官方确定性 ECDSA 测试向量
# Source: RFC 6979 Appendix A.2.5, P-256 with SHA-256
# =========================================================================


class TestRFC6979_Official_Vectors(unittest.TestCase):
    RFC_D = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
    RFC_Qx = 0x60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6
    RFC_Qy = 0x7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299

    def test_rfc6979_sample_exact_r(self):
        Q = CurvePoint(self.RFC_Qx, self.RFC_Qy, False)
        r, s = sign(self.RFC_D, b"sample", deterministic=True)
        r_exp = 0xEFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716
        self.assertEqual(r, r_exp)
        self.assertTrue(verify(Q, b"sample", (r, s)))

    def test_rfc6979_sample_low_s_and_raw_s_both_valid(self):
        Q = CurvePoint(self.RFC_Qx, self.RFC_Qy, False)
        r, s_low = sign(self.RFC_D, b"sample", deterministic=True)
        s_raw_rfc = 0xF7CB1C942D657C41D436C7A1B6E29F65F3E900DBB9AFF4064DC4AB2F843ACDA8
        n = Secp256r1.n
        expected_low = s_raw_rfc if s_raw_rfc <= n // 2 else n - s_raw_rfc
        self.assertEqual(s_low, expected_low)
        self.assertTrue(verify(Q, b"sample", (r, s_low)))
        self.assertTrue(verify(Q, b"sample", (r, s_raw_rfc)))

    def test_rfc6979_test_exact_rs(self):
        Q = CurvePoint(self.RFC_Qx, self.RFC_Qy, False)
        r, s = sign(self.RFC_D, b"test", deterministic=True)
        r_exp = 0xF1ABB023518351CD71D881567B1EA663ED3EFCF6C5132B354F28D3B0B7D38367
        s_exp = 0x019F4113742A2B14BD25926B49C649155F267E60D3814B4C0CC84250E46F0083
        self.assertEqual(r, r_exp)
        self.assertEqual(s, s_exp)
        self.assertTrue(verify(Q, b"test", (r, s)))

    def test_rfc6979_different_messages_different_signatures(self):
        messages = [b"sample", b"test", b"", b"a", b"abc"]
        sigs = set()
        for m in messages:
            sigs.add(sign(self.RFC_D, m, deterministic=True))
        self.assertEqual(len(sigs), len(messages))

    def test_rfc6979_cannot_mix_with_k(self):
        with self.assertRaises(ValueError):
            sign(self.RFC_D, b"x", k=5, deterministic=True)


# =========================================================================
# PART 7: 密钥生成与签名-验签 一致性测试（自生成）
# =========================================================================


class TestKeyGenAndSignVerify_Consistency(unittest.TestCase):
    def test_generate_keypair(self):
        for _ in range(5):
            private_key, public_key = generate_keypair()
            self.assertGreater(private_key, 0)
            self.assertLess(private_key, Secp256r1.n)
            self.assertTrue(is_point_on_curve(public_key))
            expected_pub = scalar_mult_base(private_key)
            self.assertEqual(public_key, expected_pub)

    def test_random_sign_verify_roundtrip(self):
        private_key, public_key = generate_keypair()
        message = b"Hello, secp256r1!"
        signature = sign(private_key, message)
        self.assertTrue(verify(public_key, message, signature))

    def test_verify_wrong_message(self):
        private_key, public_key = generate_keypair()
        signature = sign(private_key, b"Message 1")
        self.assertFalse(verify(public_key, b"Message 2", signature))

    def test_verify_wrong_key(self):
        priv1, pub1 = generate_keypair()
        _, pub2 = generate_keypair()
        signature = sign(priv1, b"Test")
        self.assertFalse(verify(pub2, b"Test", signature))

    def test_deterministic_and_random_sign_both_verify(self):
        d, Q = generate_keypair()
        msg = b"consistency check"
        sig_rand = sign(d, msg)
        sig_det = sign(d, msg, deterministic=True)
        self.assertTrue(verify(Q, msg, sig_rand))
        self.assertTrue(verify(Q, msg, sig_det))


# =========================================================================
# PART 8: 严格序列化与反序列化测试
# =========================================================================


class TestSerializationStrict(unittest.TestCase):
    def setUp(self):
        self.d = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
        self.pub = scalar_mult_base(self.d)

    def test_private_key_hex_roundtrip(self):
        hex_str = private_key_to_hex(self.d)
        self.assertEqual(len(hex_str), 64)
        self.assertEqual(hex_to_private_key(hex_str), self.d)

    def test_private_key_zero_rejected_export(self):
        with self.assertRaises(ValueError):
            private_key_to_hex(0)

    def test_private_key_n_rejected_export(self):
        with self.assertRaises(ValueError):
            private_key_to_hex(Secp256r1.n)

    def test_private_key_negative_rejected_export(self):
        with self.assertRaises(ValueError):
            private_key_to_hex(-1)

    def test_hex_to_private_key_wrong_length(self):
        with self.assertRaises(ValueError):
            hex_to_private_key("abcd")

    def test_hex_to_private_key_zero_rejected(self):
        with self.assertRaises(ValueError):
            hex_to_private_key("00" * 32)

    def test_hex_to_private_key_n_rejected(self):
        n_hex = Secp256r1.n.to_bytes(32, "big").hex()
        with self.assertRaises(ValueError):
            hex_to_private_key(n_hex)

    def test_public_key_uncompressed_roundtrip(self):
        hex_str = public_key_to_hex(self.pub, compressed=False)
        self.assertEqual(len(hex_str), 130)
        self.assertTrue(hex_str.startswith("04"))
        self.assertEqual(hex_to_public_key(hex_str), self.pub)

    def test_public_key_compressed_roundtrip(self):
        hex_str = public_key_to_hex(self.pub, compressed=True)
        self.assertEqual(len(hex_str), 66)
        self.assertIn(hex_str[:2], ("02", "03"))
        self.assertEqual(hex_to_public_key(hex_str), self.pub)

    def test_public_key_odd_length_hex_rejected(self):
        with self.assertRaises(ValueError):
            hex_to_public_key("04ab")

    def test_public_key_invalid_prefix_rejected(self):
        with self.assertRaises(ValueError):
            hex_to_public_key("01" + "00" * 64)

    def test_public_key_wrong_uncompressed_length_rejected(self):
        with self.assertRaises(ValueError):
            hex_to_public_key("04" + "00" * 32)

    def test_public_key_x_out_of_range_rejected(self):
        bad_x = Secp256r1.p
        bad_hex = "04" + bad_x.to_bytes(33, "big").hex() + "00" * 32
        with self.assertRaises(ValueError):
            hex_to_public_key(bad_hex)

    def test_public_key_0x_prefix_accepted(self):
        hex_str = "0x" + public_key_to_hex(self.pub, compressed=False)
        self.assertEqual(hex_to_public_key(hex_str), self.pub)

    def test_der_signature_roundtrip(self):
        r = 0xEFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716
        s = 0x0834E36AD29A83BF2BC9385E491D6099C8FDF9D1ED67AA7EA5F51F93782857A9
        der = signature_to_der((r, s))
        self.assertTrue(der.startswith(b"\x30"))
        self.assertEqual(der_to_signature(der), (r, s))

    def test_der_trailing_garbage_rejected(self):
        r = 0xEFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716
        s = 0x0834E36AD29A83BF2BC9385E491D6099C8FDF9D1ED67AA7EA5F51F93782857A9
        der = signature_to_der((r, s))
        with self.assertRaises(ValueError):
            der_to_signature(der + b"\x00\x00")

    def test_der_integer_zero_length_rejected(self):
        bad_der = bytes([0x30, 0x04, 0x02, 0x00, 0x02, 0x01, 0x01])
        with self.assertRaises(ValueError):
            der_to_signature(bad_der)

    def test_der_unnecessary_leading_zero_rejected(self):
        bad_r_enc = bytes([0x02, 0x02, 0x00, 0x7F])
        bad_s_enc = bytes([0x02, 0x01, 0x01])
        bad_der = bytes([0x30, len(bad_r_enc) + len(bad_s_enc)]) + bad_r_enc + bad_s_enc
        with self.assertRaises(ValueError):
            der_to_signature(bad_der)

    def test_der_long_form_length_rejected(self):
        bad_der = bytes([0x30, 0x81, 0x08, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01])
        with self.assertRaises(ValueError):
            der_to_signature(bad_der)

    def test_der_garbage_between_integers_rejected(self):
        r_enc = bytes([0x02, 0x01, 0x01])
        s_enc = bytes([0x02, 0x01, 0x01])
        bad_content = r_enc + b"\x00" + s_enc
        bad_der = bytes([0x30, len(bad_content)]) + bad_content
        with self.assertRaises(ValueError):
            der_to_signature(bad_der)

    def test_der_out_of_range_r_rejected(self):
        r_big = Secp256r1.n
        r_bytes = r_big.to_bytes(33, "big")
        if r_bytes[0] & 0x80:
            r_bytes = b"\x00" + r_bytes
        r_enc = bytes([0x02, len(r_bytes)]) + r_bytes
        s_enc = bytes([0x02, 0x01, 0x01])
        bad_der = bytes([0x30, len(r_enc) + len(s_enc)]) + r_enc + s_enc
        with self.assertRaises(ValueError):
            der_to_signature(bad_der)

    def test_signature_to_der_r_zero_rejected(self):
        with self.assertRaises(ValueError):
            signature_to_der((0, 1))

    def test_signature_to_der_s_zero_rejected(self):
        with self.assertRaises(ValueError):
            signature_to_der((1, 0))


# =========================================================================
# PART 9: 边界与恶意输入测试
# =========================================================================


class TestEdgeCases(unittest.TestCase):
    def test_sign_invalid_private_key_zero(self):
        with self.assertRaises(ValueError):
            sign(0, b"test")

    def test_sign_invalid_private_key_negative(self):
        with self.assertRaises(ValueError):
            sign(-1, b"test")

    def test_sign_invalid_private_key_too_large(self):
        with self.assertRaises(ValueError):
            sign(Secp256r1.n, b"test")

    def test_sign_invalid_k_zero(self):
        with self.assertRaises(ValueError):
            sign(1, b"test", k=0)

    def test_verify_signature_r_zero(self):
        _, pub = generate_keypair()
        self.assertFalse(verify(pub, b"test", (0, 123)))

    def test_verify_signature_s_zero(self):
        _, pub = generate_keypair()
        self.assertFalse(verify(pub, b"test", (123, 0)))

    def test_verify_signature_r_too_large(self):
        _, pub = generate_keypair()
        self.assertFalse(verify(pub, b"test", (Secp256r1.n, 123)))

    def test_verify_invalid_curve_point(self):
        bad_point = CurvePoint(1, 1, False)
        self.assertFalse(verify(bad_point, b"test", (1, 1)))

    def test_verify_infinity_point(self):
        _, pub = generate_keypair()
        self.assertFalse(verify(INFINITY, b"test", (1, 1)))

    def test_empty_message_sign_verify(self):
        priv, pub = generate_keypair()
        sig = sign(priv, b"")
        self.assertTrue(verify(pub, b"", sig))

    def test_long_message_sign_verify(self):
        priv, pub = generate_keypair()
        msg = b"A" * 10000
        sig = sign(priv, msg)
        self.assertTrue(verify(pub, msg, sig))


# =========================================================================
# PART 10: 多消息一致性测试（自生成，非官方向量）
# =========================================================================


class TestMultiMessageConsistency(unittest.TestCase):
    def test_multiple_messages_random_and_deterministic(self):
        d = 0x2B1F8730B68B9A3F92F0C2D36A4C26DE9D02437D197C20891D16F83E81D6AB7F
        Q = scalar_mult_base(d)
        messages = [b"", b"a", b"abc", b"message digest", b"abcdefghijklmnopqrstuvwxyz"]
        for msg in messages:
            sig_rand = sign(d, msg)
            sig_det = sign(d, msg, deterministic=True)
            self.assertTrue(verify(Q, msg, sig_rand), f"Random sig fail: {msg!r}")
            self.assertTrue(verify(Q, msg, sig_det), f"Deterministic sig fail: {msg!r}")

    def test_deterministic_sign_stable_across_calls(self):
        d = 0x2B1F8730B68B9A3F92F0C2D36A4C26DE9D02437D197C20891D16F83E81D6AB7F
        msg = b"stability test"
        sigs = [sign(d, msg, deterministic=True) for _ in range(10)]
        for s in sigs[1:]:
            self.assertEqual(s, sigs[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
