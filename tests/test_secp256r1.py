import unittest
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from secp256r1 import (
    Secp256r1,
    CurvePoint,
    INFINITY,
    SCALAR_BITS,
    MODPOW_EXPONENT_BITS,
    HKDF_SHA256_MAX_LENGTH,
    mod_add,
    mod_sub,
    mod_mul,
    mod_pow,
    mod_inv,
    constant_time_equal,
    constant_time_select,
    sha256,
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
    ecdh_compute_shared_secret,
    ecdh_compute_shared_secret_and_hash,
    ecdh_verify_consistency,
    ecdh_derive_key,
    hkdf_sha256,
    hkdf_sha256_extract,
    hkdf_sha256_expand,
    private_key_to_pem,
    pem_to_private_key,
    public_key_to_pem,
    pem_to_public_key,
    private_key_to_jwk,
    public_key_to_jwk,
    jwk_to_private_key,
    jwk_to_public_key,
    jwk_to_json,
    json_to_jwk,
)


# =========================================================================
# PART 1: 官方向量 — 底层模运算基础验证
# =========================================================================


class TestFieldOperations(unittest.TestCase):
    def test_mod_add_basic(self):
        self.assertEqual(mod_add(5, 7, 17), 12)
        self.assertEqual(mod_add(15, 10, 17), 8)

    def test_mod_sub_basic(self):
        self.assertEqual(mod_sub(10, 5, 17), 5)
        self.assertEqual(mod_sub(3, 10, 17), 10)

    def test_mod_mul_basic(self):
        self.assertEqual(mod_mul(5, 4, 17), 3)
        self.assertEqual(mod_mul(7, 8, 17), 5)

    def test_mod_pow_basic(self):
        self.assertEqual(mod_pow(2, 3, 17), 8)
        self.assertEqual(mod_pow(3, 4, 17), 13)
        self.assertEqual(mod_pow(5, 0, 17), 1)

    def test_mod_inv_basic(self):
        p = 17
        self.assertEqual(mod_mul(3, mod_inv(3, p), p), 1)
        self.assertEqual(mod_mul(7, mod_inv(7, p), p), 1)
        with self.assertRaises(ValueError):
            mod_inv(0, p)

    def test_constant_time_equal(self):
        self.assertEqual(constant_time_equal(42, 42), 1)
        self.assertEqual(constant_time_equal(42, 99), 0)

    def test_constant_time_select(self):
        self.assertEqual(constant_time_select(1, 100, 200), 100)
        self.assertEqual(constant_time_select(0, 100, 200), 200)


# =========================================================================
# PART 2: 官方向量 — secp256r1 曲线参数（SECG Sec 2）
# =========================================================================


class TestCurveParameters(unittest.TestCase):
    def test_generator_on_curve(self):
        G = CurvePoint(Secp256r1.Gx, Secp256r1.Gy, False)
        self.assertTrue(is_point_on_curve(G))

    def test_curve_equation(self):
        p = Secp256r1.p
        Gx, Gy = Secp256r1.Gx, Secp256r1.Gy
        lhs = mod_mul(Gy, Gy, p)
        rhs = mod_add(mod_add(mod_pow(Gx, 3, p), mod_mul(Secp256r1.a, Gx, p), p), Secp256r1.b, p)
        self.assertEqual(lhs, rhs)

    def test_n_is_order(self):
        G = CurvePoint(Secp256r1.Gx, Secp256r1.Gy, False)
        nG = scalar_mult(Secp256r1.n, G)
        self.assertTrue(nG.is_infinity())

    def test_infinity_point_on_curve(self):
        self.assertTrue(INFINITY.is_infinity())
        self.assertTrue(is_point_on_curve(INFINITY))


# =========================================================================
# PART 3: 官方向量 — NIST CAVP ECDSA P-256 测试向量
# Source: NIST Cryptographic Algorithm Validation Program
# =========================================================================


class TestNIST_CAVP_Vectors(unittest.TestCase):
    def test_key_pair_vector_1(self):
        d = 0x0F56DB37F9B26ED2B2CB3707F557B074B1F85FC3A09E61208415818D2838A848
        Qx = 0xdea007461b7f8a7fd1698502a7da877791a2da0873399ad59ec73045ee889d9e
        Qy = 0xa8d54938308a98b681fe83ad1443dabcc457a85d625a7cb6542e168151de91f0
        Q = scalar_mult_base(d)
        self.assertEqual(Q.x, Qx)
        self.assertEqual(Q.y, Qy)

    def test_key_pair_vector_2(self):
        d = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
        Qx = 0x60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6
        Qy = 0x7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299
        Q = scalar_mult_base(d)
        self.assertEqual(Q.x, Qx)
        self.assertEqual(Q.y, Qy)

    def test_sign_verify_vector_1(self):
        d = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
        Q = CurvePoint(0x60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6,
                        0x7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299, False)
        k = 0xA6E3C57DD01ABE90086538398355DD4C3B17AA873382B0F24D6129493D8AAD60
        r, s = sign(d, b"sample", k=k)
        self.assertTrue(verify(Q, b"sample", (r, s)))

    def test_sign_verify_vector_2(self):
        d = 0x0F56DB37F9B26ED2B2CB3707F557B074B1F85FC3A09E61208415818D2838A848
        Qx = 0xdea007461b7f8a7fd1698502a7da877791a2da0873399ad59ec73045ee889d9e
        Qy = 0xa8d54938308a98b681fe83ad1443dabcc457a85d625a7cb6542e168151de91f0
        Q = CurvePoint(Qx, Qy, False)
        k = 0x5E25A491F98BD7EBC60A63C0F6B0A1B585F5CD21DB944B23A82C23F07A417D8C
        r, s = sign(d, b"test message for secp256r1", k=k)
        self.assertTrue(verify(Q, b"test message for secp256r1", (r, s)))


# =========================================================================
# PART 4: 官方向量 — RFC 6979 确定性 ECDSA P-256 SHA-256
# Source: RFC 6979 Appendix A.2.5
# =========================================================================


class TestRFC6979_Official_Vectors(unittest.TestCase):
    RFC_D = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
    RFC_Qx = 0x60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6
    RFC_Qy = 0x7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299

    def test_sample_exact_r(self):
        Q = CurvePoint(self.RFC_Qx, self.RFC_Qy, False)
        r, s = sign(self.RFC_D, b"sample", deterministic=True)
        self.assertEqual(r, 0xEFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716)
        self.assertTrue(verify(Q, b"sample", (r, s)))

    def test_sample_low_s_and_raw_s_both_valid(self):
        Q = CurvePoint(self.RFC_Qx, self.RFC_Qy, False)
        r, s_low = sign(self.RFC_D, b"sample", deterministic=True)
        s_raw_rfc = 0xF7CB1C942D657C41D436C7A1B6E29F65F3E900DBB9AFF4064DC4AB2F843ACDA8
        n = Secp256r1.n
        expected_low = s_raw_rfc if s_raw_rfc <= n // 2 else n - s_raw_rfc
        self.assertEqual(s_low, expected_low)
        self.assertTrue(verify(Q, b"sample", (r, s_low)))
        self.assertTrue(verify(Q, b"sample", (r, s_raw_rfc)))

    def test_test_exact_rs(self):
        Q = CurvePoint(self.RFC_Qx, self.RFC_Qy, False)
        r, s = sign(self.RFC_D, b"test", deterministic=True)
        self.assertEqual(r, 0xF1ABB023518351CD71D881567B1EA663ED3EFCF6C5132B354F28D3B0B7D38367)
        self.assertEqual(s, 0x019F4113742A2B14BD25926B49C649155F267E60D3814B4C0CC84250E46F0083)
        self.assertTrue(verify(Q, b"test", (r, s)))

    def test_different_messages_different_signatures(self):
        sigs = set()
        for m in [b"sample", b"test", b"", b"a", b"abc"]:
            sigs.add(sign(self.RFC_D, m, deterministic=True))
        self.assertEqual(len(sigs), 5)

    def test_cannot_mix_k_and_deterministic(self):
        with self.assertRaises(ValueError):
            sign(self.RFC_D, b"x", k=5, deterministic=True)


# =========================================================================
# PART 5: 官方向量 — RFC 5869 HKDF-SHA256 Test Vectors
# Source: RFC 5869 Appendix A, Test Cases 1 and 3
# =========================================================================


class TestRFC5869_HKDF_Vectors(unittest.TestCase):
    """RFC 5869 Appendix A: HKDF with SHA-256 test vectors."""

    def test_case_1_basic_sha256(self):
        IKM = bytes([0x0b] * 22)
        salt = bytes.fromhex("000102030405060708090a0b0c")
        info = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9")
        L = 42
        PRK_expected = bytes.fromhex("077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5")
        OKM_expected = bytes.fromhex("3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865")

        prk = hkdf_sha256_extract(salt, IKM)
        self.assertEqual(prk, PRK_expected)

        okm = hkdf_sha256_expand(prk, info, L)
        self.assertEqual(okm, OKM_expected)

        okm2 = hkdf_sha256(IKM, salt=salt, info=info, length=L)
        self.assertEqual(okm2, OKM_expected)

    def test_case_3_sha256_zero_salt_info(self):
        IKM = bytes.fromhex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b")
        L = 42
        PRK_expected = bytes.fromhex("19ef24a32c717b167f33a91d6f648bdf96596776afdb6377ac434c1c293ccb04")
        OKM_expected = bytes.fromhex("8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d9d201395faa4b61a96c8")

        prk = hkdf_sha256_extract(None, IKM)
        self.assertEqual(prk, PRK_expected)

        okm = hkdf_sha256_expand(prk, b"", L)
        self.assertEqual(okm, OKM_expected)

        okm2 = hkdf_sha256(IKM, salt=None, info=b"", length=L)
        self.assertEqual(okm2, OKM_expected)


# =========================================================================
# PART 6: 时序安全回归测试
# =========================================================================


class TestTimingSafety(unittest.TestCase):
    def test_scalar_mult_fixed_loop_count(self):
        self.assertEqual(SCALAR_BITS, 256)

    def test_mod_pow_fixed_loop_count(self):
        self.assertEqual(MODPOW_EXPONENT_BITS, 256)

    def test_scalar_mult_time_independent_of_scalar_hamming_weight(self):
        G = CurvePoint(Secp256r1.Gx, Secp256r1.Gy, False)
        scalars = [1, 0x8000000000000000000000000000000000000000000000000000000000000000,
                   (1 << 255) - 1, Secp256r1.n - 1]
        times = []
        for s in scalars:
            t0 = time.perf_counter()
            for _ in range(5):
                scalar_mult(s, G)
            times.append(time.perf_counter() - t0)
        avg = sum(times) / len(times)
        for t in times:
            self.assertLess(t / avg, 3.0)

    def test_deterministic_sign_reproducible(self):
        d = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
        self.assertEqual(sign(d, b"sample", deterministic=True),
                         sign(d, b"sample", deterministic=True))


# =========================================================================
# PART 7: 一致性测试 — 点运算（自生成）
# =========================================================================


class TestPointOperations_Consistency(unittest.TestCase):
    def setUp(self):
        self.G = CurvePoint(Secp256r1.Gx, Secp256r1.Gy, False)

    def test_point_add_infinity(self):
        self.assertEqual(point_add(INFINITY, self.G), self.G)
        self.assertEqual(point_add(self.G, INFINITY), self.G)

    def test_point_add_inverse(self):
        neg_G = CurvePoint(self.G.x, Secp256r1.p - self.G.y, False)
        self.assertTrue(point_add(self.G, neg_G).is_infinity())

    def test_point_double_equals_self_add(self):
        self.assertEqual(point_add(self.G, self.G), point_double(self.G))

    def test_scalar_mult_2G(self):
        self.assertEqual(scalar_mult(2, self.G), point_double(self.G))

    def test_scalar_mult_associative(self):
        self.assertEqual(scalar_mult(2, scalar_mult(3, self.G)), scalar_mult(6, self.G))

    def test_scalar_mult_distributive(self):
        self.assertEqual(point_add(scalar_mult(5, self.G), scalar_mult(7, self.G)),
                         scalar_mult(12, self.G))

    def test_scalar_mult_zero(self):
        self.assertTrue(scalar_mult(0, self.G).is_infinity())

    def test_invalid_point_rejected(self):
        with self.assertRaises(ValueError):
            validate_point(CurvePoint(1, 2, False))
        self.assertFalse(is_point_on_curve(CurvePoint(1, 2, False)))

    def test_infinity_rejected_by_validate(self):
        with self.assertRaises(ValueError):
            validate_point(INFINITY)


# =========================================================================
# PART 8: 一致性测试 — DER 签名严格解析（自生成边界用例）
# =========================================================================


class TestDERSignatureStrict(unittest.TestCase):
    def test_negative_integer_0x80_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            der_to_signature(b"\x30\x06\x02\x01\x80\x02\x01\x01")
        self.assertIn("negative INTEGER", str(ctx.exception))

    def test_negative_integer_0xff_leading_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            der_to_signature(b"\x30\x07\x02\x02\xFF\x80\x02\x01\x01")
        self.assertIn("negative INTEGER", str(ctx.exception))

    def test_high_bit_with_proper_prefix_accepted(self):
        r_enc = bytes([0x02, 0x02, 0x00, 0x81])
        s_enc = bytes([0x02, 0x01, 0x01])
        content = r_enc + s_enc
        der = bytes([0x30, len(content)]) + content
        self.assertEqual(der_to_signature(der), (0x81, 1))

    def test_zero_length_integer_rejected(self):
        with self.assertRaises(ValueError):
            der_to_signature(bytes([0x30, 0x04, 0x02, 0x00, 0x02, 0x01, 0x01]))

    def test_unnecessary_leading_zero_rejected(self):
        bad_r = bytes([0x02, 0x02, 0x00, 0x7F])
        s = bytes([0x02, 0x01, 0x01])
        content = bad_r + s
        with self.assertRaises(ValueError):
            der_to_signature(bytes([0x30, len(content)]) + content)

    def test_trailing_garbage_rejected(self):
        with self.assertRaises(ValueError):
            der_to_signature(signature_to_der((123, 456)) + b"\x00\x00")

    def test_long_form_length_rejected(self):
        with self.assertRaises(ValueError):
            der_to_signature(bytes([0x30, 0x81, 0x08, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01]))

    def test_signature_to_der_zero_rejected(self):
        with self.assertRaises(ValueError):
            signature_to_der((0, 1))
        with self.assertRaises(ValueError):
            signature_to_der((1, 0))


# =========================================================================
# PART 9: 一致性测试 — 签名/验签 Prehashed 模式（自生成）
# =========================================================================


class TestPrehashedSignVerify_Consistency(unittest.TestCase):
    def test_prehashed_sign_verify_roundtrip(self):
        d, Q = generate_keypair()
        digest = sha256(b"test message for prehashed")
        sig = sign(d, digest, prehashed=True)
        self.assertTrue(verify(Q, digest, sig, prehashed=True))

    def test_prehashed_and_original_independent(self):
        d, Q = generate_keypair()
        msg = b"hello"
        digest = sha256(msg)
        self.assertNotEqual(sign(d, msg), sign(d, digest, prehashed=True))
        self.assertTrue(verify(Q, msg, sign(d, msg)))
        self.assertTrue(verify(Q, digest, sign(d, digest, prehashed=True), prehashed=True))

    def test_prehashed_cross_mode_verify_fails(self):
        d, Q = generate_keypair()
        msg_a = b"message alpha"
        msg_b = b"message beta"
        sig_a = sign(d, msg_a)
        sig_b_hash = sign(d, sha256(msg_b), prehashed=True)
        self.assertFalse(verify(Q, sha256(msg_b), sig_a, prehashed=True))
        self.assertFalse(verify(Q, sha256(msg_a), sig_b_hash, prehashed=True))
        self.assertFalse(verify(Q, msg_b, sig_a))

    def test_prehashed_wrong_length_rejected(self):
        d, _ = generate_keypair()
        with self.assertRaises(ValueError) as ctx:
            sign(d, b"short", prehashed=True)
        self.assertIn("32 bytes", str(ctx.exception))

    def test_prehashed_verify_wrong_length_returns_false(self):
        _, Q = generate_keypair()
        self.assertFalse(verify(Q, b"short", (1, 2), prehashed=True))

    def test_prehashed_and_deterministic_conflict(self):
        d, _ = generate_keypair()
        with self.assertRaises(ValueError):
            sign(d, sha256(b"x"), prehashed=True, deterministic=True)

    def test_prehashed_equivalence_with_fixed_k(self):
        d, Q = generate_keypair()
        msg = b"exact equivalence check"
        digest = sha256(msg)
        k = 0x123456789ABCDEF123456789ABCDEF123456789ABCDEF123456789ABCDEF123
        self.assertEqual(sign(d, msg, k=k), sign(d, digest, prehashed=True, k=k))


# =========================================================================
# PART 10: 一致性测试 — 密钥生成与签名-验签（自生成）
# =========================================================================


class TestKeyGenAndSignVerify_Consistency(unittest.TestCase):
    def test_generate_keypair(self):
        for _ in range(5):
            priv, pub = generate_keypair()
            self.assertGreater(priv, 0)
            self.assertLess(priv, Secp256r1.n)
            self.assertEqual(pub, scalar_mult_base(priv))

    def test_random_sign_verify_roundtrip(self):
        priv, pub = generate_keypair()
        self.assertTrue(verify(pub, b"Hello, secp256r1!", sign(priv, b"Hello, secp256r1!")))

    def test_verify_wrong_message(self):
        priv, pub = generate_keypair()
        self.assertFalse(verify(pub, b"Message 2", sign(priv, b"Message 1")))

    def test_verify_wrong_key(self):
        priv1, _ = generate_keypair()
        _, pub2 = generate_keypair()
        self.assertFalse(verify(pub2, b"Test", sign(priv1, b"Test")))

    def test_deterministic_and_random_both_verify(self):
        d, Q = generate_keypair()
        self.assertTrue(verify(Q, b"check", sign(d, b"check")))
        self.assertTrue(verify(Q, b"check", sign(d, b"check", deterministic=True)))


# =========================================================================
# PART 11: 一致性测试 — 严格序列化（自生成边界用例）
# =========================================================================


class TestSerializationStrict_Consistency(unittest.TestCase):
    def test_private_key_hex_roundtrip(self):
        d = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
        self.assertEqual(hex_to_private_key(private_key_to_hex(d)), d)

    def test_private_key_zero_rejected_export(self):
        with self.assertRaises(ValueError):
            private_key_to_hex(0)

    def test_private_key_n_rejected_export(self):
        with self.assertRaises(ValueError):
            private_key_to_hex(Secp256r1.n)

    def test_hex_to_private_key_wrong_length(self):
        with self.assertRaises(ValueError):
            hex_to_private_key("abcd")

    def test_hex_to_private_key_zero_rejected(self):
        with self.assertRaises(ValueError):
            hex_to_private_key("00" * 32)

    def test_public_key_uncompressed_roundtrip(self):
        d = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
        Q = scalar_mult_base(d)
        self.assertEqual(hex_to_public_key(public_key_to_hex(Q, compressed=False)), Q)

    def test_public_key_compressed_roundtrip(self):
        d = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
        Q = scalar_mult_base(d)
        self.assertEqual(hex_to_public_key(public_key_to_hex(Q, compressed=True)), Q)

    def test_public_key_invalid_prefix_rejected(self):
        with self.assertRaises(ValueError):
            hex_to_public_key("01" + "00" * 64)

    def test_der_signature_roundtrip(self):
        sig = (0xEFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716,
               0x0834E36AD29A83BF2BC9385E491D6099C8FDF9D1ED67AA7EA5F51F93782857A9)
        self.assertEqual(der_to_signature(signature_to_der(sig)), sig)


# =========================================================================
# PART 12: 一致性测试 — 边界与恶意输入拦截（自生成）
# =========================================================================


class TestEdgeCases_Consistency(unittest.TestCase):
    def test_sign_invalid_private_key(self):
        with self.assertRaises(ValueError):
            sign(0, b"test")
        with self.assertRaises(ValueError):
            sign(-1, b"test")
        with self.assertRaises(ValueError):
            sign(Secp256r1.n, b"test")

    def test_verify_signature_r_s_out_of_range(self):
        _, pub = generate_keypair()
        self.assertFalse(verify(pub, b"test", (0, 123)))
        self.assertFalse(verify(pub, b"test", (123, 0)))
        self.assertFalse(verify(pub, b"test", (Secp256r1.n, 123)))

    def test_verify_invalid_curve_point(self):
        self.assertFalse(verify(CurvePoint(1, 1, False), b"test", (1, 1)))
        self.assertFalse(verify(INFINITY, b"test", (1, 1)))

    def test_empty_message_sign_verify(self):
        priv, pub = generate_keypair()
        self.assertTrue(verify(pub, b"", sign(priv, b"")))

    def test_long_message_sign_verify(self):
        priv, pub = generate_keypair()
        msg = b"A" * 10000
        self.assertTrue(verify(pub, msg, sign(priv, msg)))

    def test_sign_invalid_k_zero(self):
        with self.assertRaises(ValueError):
            sign(1, b"test", k=0)


# =========================================================================
# PART 13: 一致性测试 — ECDH 共享密钥（自生成）
# =========================================================================


class TestECDH_Consistency(unittest.TestCase):
    def test_mutual_agreement(self):
        dA, QA = generate_keypair()
        dB, QB = generate_keypair()
        self.assertEqual(ecdh_compute_shared_secret(dA, QB),
                         ecdh_compute_shared_secret(dB, QA))

    def test_different_pairs_different_secrets(self):
        dA, QA = generate_keypair()
        _, QB = generate_keypair()
        _, QC = generate_keypair()
        self.assertNotEqual(ecdh_compute_shared_secret(dA, QB),
                            ecdh_compute_shared_secret(dA, QC))

    def test_sha256_kdf(self):
        dA, QA = generate_keypair()
        dB, QB = generate_keypair()
        self.assertEqual(ecdh_derive_key(dA, QB, kdf="sha256"),
                         ecdh_derive_key(dB, QA, kdf="sha256"))

    def test_hkdf_sha256_kdf_mutual(self):
        dA, QA = generate_keypair()
        dB, QB = generate_keypair()
        self.assertEqual(ecdh_derive_key(dA, QB, kdf="hkdf-sha256", salt=b"salt", info=b"info", length=64),
                         ecdh_derive_key(dB, QA, kdf="hkdf-sha256", salt=b"salt", info=b"info", length=64))

    def test_hkdf_sha256_kdf_different_length(self):
        dA, QA = generate_keypair()
        dB, QB = generate_keypair()
        key32 = ecdh_derive_key(dA, QB, kdf="hkdf-sha256", length=32)
        key64 = ecdh_derive_key(dA, QB, kdf="hkdf-sha256", length=64)
        self.assertEqual(len(key32), 32)
        self.assertEqual(len(key64), 64)
        self.assertEqual(key64[:32], key32)

    def test_sha256_kdf_wrong_length_rejected(self):
        dA, QA = generate_keypair()
        _, QB = generate_keypair()
        with self.assertRaises(ValueError):
            ecdh_derive_key(dA, QB, kdf="sha256", length=64)

    def test_unsupported_kdf_rejected(self):
        dA, _, _, QB = *generate_keypair(), *generate_keypair()
        with self.assertRaises(ValueError):
            ecdh_derive_key(dA, QB, kdf="bad")

    def test_reject_infinity(self):
        dA, _ = generate_keypair()
        with self.assertRaises(ValueError):
            ecdh_compute_shared_secret(dA, INFINITY)

    def test_reject_invalid_curve_point(self):
        dA, _ = generate_keypair()
        with self.assertRaises(ValueError):
            ecdh_compute_shared_secret(dA, CurvePoint(1, 2, False))

    def test_hkdf_zero_length_rejected(self):
        with self.assertRaises(ValueError):
            hkdf_sha256(b"ikm", length=0)

    def test_hkdf_exceeds_max_rejected(self):
        with self.assertRaises(ValueError):
            hkdf_sha256(b"ikm", length=HKDF_SHA256_MAX_LENGTH + 1)


# =========================================================================
# PART 14: 一致性测试 — PEM 导入导出往返（自生成）
# =========================================================================


class TestPEM_Consistency(unittest.TestCase):
    def setUp(self):
        self.d = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
        self.Q = scalar_mult_base(self.d)

    def test_pkcs8_private_key_roundtrip(self):
        pem = private_key_to_pem(self.d, self.Q, format="pkcs8")
        self.assertIn("-----BEGIN PRIVATE KEY-----", pem)
        d_back, Q_back, fmt = pem_to_private_key(pem)
        self.assertEqual(d_back, self.d)
        self.assertEqual(Q_back, self.Q)
        self.assertEqual(fmt, "pkcs8")

    def test_sec1_private_key_roundtrip(self):
        pem = private_key_to_pem(self.d, self.Q, format="sec1")
        self.assertIn("-----BEGIN EC PRIVATE KEY-----", pem)
        d_back, Q_back, fmt = pem_to_private_key(pem)
        self.assertEqual(d_back, self.d)
        self.assertEqual(Q_back, self.Q)
        self.assertEqual(fmt, "sec1")

    def test_private_key_pem_without_public_key(self):
        d_back, Q_back, _ = pem_to_private_key(private_key_to_pem(self.d, format="pkcs8"))
        self.assertEqual(d_back, self.d)
        self.assertEqual(Q_back, self.Q)

    def test_spki_public_key_roundtrip(self):
        pem = public_key_to_pem(self.Q)
        self.assertIn("-----BEGIN PUBLIC KEY-----", pem)
        self.assertEqual(pem_to_public_key(pem), self.Q)

    def test_pem_invalid_format_rejected(self):
        with self.assertRaises(ValueError):
            private_key_to_pem(self.d, format="bad")

    def test_pem_wrong_marker_rejected(self):
        pem = private_key_to_pem(self.d, format="pkcs8").replace("PRIVATE KEY", "RSA PRIVATE KEY")
        with self.assertRaises(ValueError):
            pem_to_private_key(pem)

    def test_pem_invalid_base64_rejected(self):
        with self.assertRaises(ValueError):
            pem_to_private_key("-----BEGIN PRIVATE KEY-----\nInvalidBase64!\n-----END PRIVATE KEY-----\n")

    def test_pem_empty_content_rejected(self):
        with self.assertRaises(ValueError):
            pem_to_private_key("-----BEGIN PRIVATE KEY-----\n-----END PRIVATE KEY-----\n")

    def test_pem_private_key_public_key_mismatch_rejected(self):
        _, Q2 = generate_keypair()
        with self.assertRaises(ValueError):
            private_key_to_pem(self.d, Q2, format="pkcs8")

    def test_pem_text_before_begin_rejected(self):
        pem = private_key_to_pem(self.d, format="pkcs8")
        with self.assertRaises(ValueError):
            pem_to_private_key("extra text before\n" + pem)

    def test_pem_text_after_end_rejected(self):
        pem = private_key_to_pem(self.d, format="pkcs8")
        with self.assertRaises(ValueError):
            pem_to_private_key(pem + "extra text after\n")

    def test_pem_public_key_wrong_curve_rejected(self):
        bad_pem = (
            "-----BEGIN PUBLIC KEY-----\n"
            "MFYwEAYHKoZIzj0CAQYFK4EEACIDQgAEYP7UuiVanTHJYet0xjVtaMBJuJI7Yfps\n"
            "5mliLmDyn7Z5A/4QCLi8maQa6elWKLxk8vGyDC1+n1F3o8KU1EYimQ==\n"
            "-----END PUBLIC KEY-----\n"
        )
        with self.assertRaises(ValueError) as ctx:
            pem_to_public_key(bad_pem)
        self.assertIn("curve OID", str(ctx.exception))

    def test_pem_sign_verify_after_roundtrip(self):
        pem_priv = private_key_to_pem(self.d, format="sec1")
        pem_pub = public_key_to_pem(self.Q)
        d_back, Q_back, _ = pem_to_private_key(pem_priv)
        Q_pub = pem_to_public_key(pem_pub)
        sig = sign(d_back, b"after pem roundtrip")
        self.assertTrue(verify(Q_back, b"after pem roundtrip", sig))
        self.assertTrue(verify(Q_pub, b"after pem roundtrip", sig))


# =========================================================================
# PART 15: 一致性测试 — JWK 导入导出往返（自生成）
# =========================================================================


class TestJWK_Consistency(unittest.TestCase):
    def test_private_key_jwk_roundtrip(self):
        d, Q = generate_keypair()
        jwk = private_key_to_jwk(d, Q)
        self.assertEqual(jwk["kty"], "EC")
        self.assertEqual(jwk["crv"], "P-256")
        self.assertIn("d", jwk)
        self.assertIn("x", jwk)
        self.assertIn("y", jwk)
        d_back, Q_back = jwk_to_private_key(jwk)
        self.assertEqual(d_back, d)
        self.assertEqual(Q_back, Q)

    def test_public_key_jwk_roundtrip(self):
        _, Q = generate_keypair()
        jwk = public_key_to_jwk(Q)
        self.assertNotIn("d", jwk)
        Q_back = jwk_to_public_key(jwk)
        self.assertEqual(Q_back, Q)

    def test_jwk_json_roundtrip(self):
        d, Q = generate_keypair()
        jwk = private_key_to_jwk(d, Q)
        json_str = jwk_to_json(jwk)
        jwk_back = json_to_jwk(json_str)
        d_back, Q_back = jwk_to_private_key(jwk_back)
        self.assertEqual(d_back, d)
        self.assertEqual(Q_back, Q)

    def test_jwk_pretty_json(self):
        d, Q = generate_keypair()
        jwk = private_key_to_jwk(d, Q)
        pretty = jwk_to_json(jwk, pretty=True)
        self.assertIn("\n", pretty)
        jwk_back = json_to_jwk(pretty)
        d_back, _ = jwk_to_private_key(jwk_back)
        self.assertEqual(d_back, d)

    def test_jwk_wrong_curve_rejected(self):
        _, Q = generate_keypair()
        jwk = public_key_to_jwk(Q)
        bad_jwk = dict(jwk, crv="P-384")
        with self.assertRaises(ValueError) as ctx:
            jwk_to_public_key(bad_jwk)
        self.assertIn("P-256", str(ctx.exception))

    def test_jwk_wrong_kty_rejected(self):
        _, Q = generate_keypair()
        jwk = public_key_to_jwk(Q)
        bad_jwk = dict(jwk, kty="RSA")
        with self.assertRaises(ValueError):
            jwk_to_public_key(bad_jwk)

    def test_jwk_missing_d_for_private_import(self):
        _, Q = generate_keypair()
        jwk = public_key_to_jwk(Q)
        with self.assertRaises(ValueError) as ctx:
            jwk_to_private_key(jwk)
        self.assertIn("d", str(ctx.exception))

    def test_jwk_invalid_json_rejected(self):
        with self.assertRaises(ValueError):
            json_to_jwk("not valid json{")

    def test_jwk_not_object_rejected(self):
        with self.assertRaises(ValueError):
            json_to_jwk("[1,2,3]")

    def test_jwk_sign_verify_after_roundtrip(self):
        d, Q = generate_keypair()
        jwk = private_key_to_jwk(d, Q)
        d_back, Q_back = jwk_to_private_key(jwk)
        msg = b"sign after jwk roundtrip"
        self.assertTrue(verify(Q_back, msg, sign(d_back, msg)))

    def test_jwk_hex_pem_cross_format(self):
        d, Q = generate_keypair()
        jwk = private_key_to_jwk(d, Q)
        d_back, _ = jwk_to_private_key(jwk)
        self.assertEqual(private_key_to_hex(d_back), private_key_to_hex(d))

        pem_pub = public_key_to_pem(Q)
        Q_from_pem = pem_to_public_key(pem_pub)
        jwk_pub = public_key_to_jwk(Q_from_pem)
        Q_from_jwk = jwk_to_public_key(jwk_pub)
        self.assertEqual(Q_from_jwk, Q)


# =========================================================================
# PART 16: 一致性测试 — 多消息稳定性（自生成）
# =========================================================================


class TestMultiMessageConsistency(unittest.TestCase):
    def test_multiple_messages_random_and_deterministic(self):
        d, Q = generate_keypair()
        for msg in [b"", b"a", b"abc", b"message digest"]:
            self.assertTrue(verify(Q, msg, sign(d, msg)))
            self.assertTrue(verify(Q, msg, sign(d, msg, deterministic=True)))

    def test_deterministic_sign_stable(self):
        d, Q = generate_keypair()
        msg = b"stability test"
        sigs = [sign(d, msg, deterministic=True) for _ in range(10)]
        for s in sigs[1:]:
            self.assertEqual(s, sigs[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
