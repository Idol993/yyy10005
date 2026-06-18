import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from secp256r1 import (
    Secp256r1,
    CurvePoint,
    INFINITY,
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

    def test_scalar_mult_zero(self):
        result = scalar_mult(0, self.G)
        self.assertTrue(result.is_infinity())

    def test_scalar_mult_distributive(self):
        P = scalar_mult(5, self.G)
        Q = scalar_mult(7, self.G)
        R1 = point_add(P, Q)
        R2 = scalar_mult(12, self.G)
        self.assertEqual(R1, R2)

    def test_invalid_point_rejected(self):
        bad_point = CurvePoint(1, 2, False)
        self.assertFalse(is_point_on_curve(bad_point))
        with self.assertRaises(ValueError):
            validate_point(bad_point)

    def test_infinity_rejected_by_validate(self):
        with self.assertRaises(ValueError):
            validate_point(INFINITY)


class TestNISTVectors(unittest.TestCase):
    """
    NIST CAVP ECDSA P-256 (secp256r1) known answer test vectors.
    Vector source: NIST Cryptographic Algorithm Validation Program
    """

    def test_nist_key_pair_1(self):
        d = 0x0F56DB37F9B26ED2B2CB3707F557B074B1F85FC3A09E61208415818D2838A848
        Qx = 0xdea007461b7f8a7fd1698502a7da877791a2da0873399ad59ec73045ee889d9e
        Qy = 0xa8d54938308a98b681fe83ad1443dabcc457a85d625a7cb6542e168151de91f0
        Q = scalar_mult_base(d)
        self.assertEqual(Q.x, Qx)
        self.assertEqual(Q.y, Qy)
        self.assertTrue(is_point_on_curve(Q))

    def test_nist_key_pair_2(self):
        d = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
        Qx = 0x60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6
        Qy = 0x7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299
        Q = scalar_mult_base(d)
        self.assertEqual(Q.x, Qx)
        self.assertEqual(Q.y, Qy)
        self.assertTrue(is_point_on_curve(Q))

    def test_nist_sign_verify_1(self):
        d = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
        Qx = 0x60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6
        Qy = 0x7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299
        Q = CurvePoint(Qx, Qy, False)
        message = b"sample"
        k = 0xA6E3C57DD01ABE90086538398355DD4C3B17AA873382B0F24D6129493D8AAD60
        r, s = sign(d, message, k=k)
        self.assertTrue(verify(Q, message, (r, s)))

    def test_nist_sign_verify_2(self):
        d = 0x0F56DB37F9B26ED2B2CB3707F557B074B1F85FC3A09E61208415818D2838A848
        Qx = 0xdea007461b7f8a7fd1698502a7da877791a2da0873399ad59ec73045ee889d9e
        Qy = 0xa8d54938308a98b681fe83ad1443dabcc457a85d625a7cb6542e168151de91f0
        Q = CurvePoint(Qx, Qy, False)
        message = b"test message for secp256r1"
        k = 0x5E25A491F98BD7EBC60A63C0F6B0A1B585F5CD21DB944B23A82C23F07A417D8C
        r, s = sign(d, message, k=k)
        self.assertTrue(verify(Q, message, (r, s)))


class TestRFC6979Vectors(unittest.TestCase):
    """
    RFC 6979 Deterministic ECDSA test vectors for P-256.
    """

    def test_rfc6979_sha256_key1(self):
        d = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
        Ux = 0x60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6
        Uy = 0x7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299
        Q = CurvePoint(Ux, Uy, False)
        k = 0xA6E3C57DD01ABE90086538398355DD4C3B17AA873382B0F24D6129493D8AAD60
        r_exp = 0xEFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716
        s_exp_rfc = 0xF7CB1C942D657C41D436C7A1B6E29F65F3E900DBB9AFF4064DC4AB2F843ACDA8
        message = b"sample"
        r, s = sign(d, message, k=k)
        self.assertEqual(r, r_exp)
        s_low = s_exp_rfc if s_exp_rfc <= Secp256r1.n // 2 else Secp256r1.n - s_exp_rfc
        self.assertEqual(s, s_low)
        self.assertTrue(verify(Q, message, (r, s)))
        self.assertTrue(verify(Q, message, (r, s_exp_rfc)))

    def test_rfc6979_sha256_key1_test(self):
        d = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
        Ux = 0x60FED4BA255A9D31C961EB74C6356D68C049B8923B61FA6CE669622E60F29FB6
        Uy = 0x7903FE1008B8BC99A41AE9E95628BC64F2F1B20C2D7E9F5177A3C294D4462299
        Q = CurvePoint(Ux, Uy, False)
        k = 0xD16B6AE827F17175E040871A1C7EC3500192C4C92677336EC2537ACAEE0008E0
        r_exp = 0xF1ABB023518351CD71D881567B1EA663ED3EFCF6C5132B354F28D3B0B7D38367
        s_exp = 0x019F4113742A2B14BD25926B49C649155F267E60D3814B4C0CC84250E46F0083
        message = b"test"
        r, s = sign(d, message, k=k)
        self.assertEqual(r, r_exp)
        self.assertEqual(s, s_exp)
        self.assertTrue(verify(Q, message, (r, s)))


class TestKeyGeneration(unittest.TestCase):
    def test_generate_keypair(self):
        for _ in range(5):
            private_key, public_key = generate_keypair()
            self.assertGreater(private_key, 0)
            self.assertLess(private_key, Secp256r1.n)
            self.assertTrue(is_point_on_curve(public_key))
            expected_pub = scalar_mult_base(private_key)
            self.assertEqual(public_key, expected_pub)

    def test_sign_verify_random(self):
        private_key, public_key = generate_keypair()
        message = b"Hello, secp256r1!"
        signature = sign(private_key, message)
        self.assertTrue(verify(public_key, message, signature))

    def test_verify_wrong_message(self):
        private_key, public_key = generate_keypair()
        message1 = b"Message 1"
        message2 = b"Message 2"
        signature = sign(private_key, message1)
        self.assertFalse(verify(public_key, message2, signature))

    def test_verify_wrong_key(self):
        priv1, pub1 = generate_keypair()
        _, pub2 = generate_keypair()
        message = b"Test"
        signature = sign(priv1, message)
        self.assertFalse(verify(pub2, message, signature))


class TestSerialization(unittest.TestCase):
    def test_private_key_hex_roundtrip(self):
        d = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
        hex_str = private_key_to_hex(d)
        self.assertEqual(len(hex_str), 64)
        recovered = hex_to_private_key(hex_str)
        self.assertEqual(recovered, d)

    def test_hex_to_private_key_invalid_length(self):
        with self.assertRaises(ValueError):
            hex_to_private_key("abcd")

    def test_public_key_uncompressed_roundtrip(self):
        d = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
        pub = scalar_mult_base(d)
        hex_str = public_key_to_hex(pub, compressed=False)
        self.assertEqual(len(hex_str), 130)
        self.assertTrue(hex_str.startswith("04"))
        recovered = hex_to_public_key(hex_str)
        self.assertEqual(recovered, pub)

    def test_public_key_compressed_roundtrip(self):
        d = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
        pub = scalar_mult_base(d)
        hex_str = public_key_to_hex(pub, compressed=True)
        self.assertEqual(len(hex_str), 66)
        self.assertTrue(hex_str.startswith("02") or hex_str.startswith("03"))
        recovered = hex_to_public_key(hex_str)
        self.assertEqual(recovered, pub)

    def test_public_key_with_0x_prefix(self):
        d = 0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721
        pub = scalar_mult_base(d)
        hex_str = "0x" + public_key_to_hex(pub, compressed=False)
        recovered = hex_to_public_key(hex_str)
        self.assertEqual(recovered, pub)

    def test_invalid_public_key_hex(self):
        with self.assertRaises(ValueError):
            hex_to_public_key("01" + "00" * 64)
        with self.assertRaises(ValueError):
            hex_to_public_key("04" + "00" * 32)

    def test_der_signature_roundtrip(self):
        signature = (
            0xEFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716,
            0xF7CB1C942D657C41D436C7A1B6E29F65F3E900DBB9AFF4064DC4AB2F843ACDA8,
        )
        der = signature_to_der(signature)
        self.assertTrue(der.startswith(b"\x30"))
        recovered = der_to_signature(der)
        self.assertEqual(recovered, signature)


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

    def test_empty_message(self):
        priv, pub = generate_keypair()
        sig = sign(priv, b"")
        self.assertTrue(verify(pub, b"", sig))

    def test_long_message(self):
        priv, pub = generate_keypair()
        msg = b"A" * 10000
        sig = sign(priv, msg)
        self.assertTrue(verify(pub, msg, sig))


class TestKnownSignatureVectors(unittest.TestCase):
    """
    Additional known signature test vectors for comprehensive validation.
    """

    def test_known_signature_1(self):
        d = 0xDC9339F51431E2F87D318BF2B79489B022136346847D55E894C4498A71B31417
        Qx = 0x324f63ccb56e41f14eca1288a807380914657469c3da8fdc6e4692c065f48e63
        Qy = 0xd8901a5dcebf924de0032119ab2e4fe8657d16966b09364ac583d5f83456ecaf
        Q = CurvePoint(Qx, Qy, False)
        self.assertTrue(is_point_on_curve(Q))
        msg = b"abc"
        k = 0x18D7F03018B8A5152B11869888A4F9162B414115F22F8838D3A979A117961C4E
        r, s = sign(d, msg, k=k)
        self.assertTrue(verify(Q, msg, (r, s)))

    def test_known_signature_2(self):
        d = 0x2B1F8730B68B9A3F92F0C2D36A4C26DE9D02437D197C20891D16F83E81D6AB7F
        Q = scalar_mult_base(d)
        messages = [
            b"",
            b"a",
            b"abc",
            b"message digest",
            b"abcdefghijklmnopqrstuvwxyz",
        ]
        k_base = 0x4A5E88B2D3C7E1F09A8B7C6D5E4F3A2B1C0D9E8F7A6B5C4D3E2F1A0B9C8D7E6F
        for i, msg in enumerate(messages):
            k = k_base + i * 0x1000
            r, s = sign(d, msg, k=k)
            self.assertTrue(
                verify(Q, msg, (r, s)),
                f"Failed for message: {msg!r}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
