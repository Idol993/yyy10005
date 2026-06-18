from .field_ops import (
    mod_add,
    mod_sub,
    mod_mul,
    mod_pow,
    mod_inv,
    constant_time_equal,
    constant_time_select,
)


class Secp256r1:
    p = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
    a = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFC
    b = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
    n = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
    h = 1
    Gx = 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296
    Gy = 0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5


class CurvePoint:
    __slots__ = ("x", "y", "infinity")

    def __init__(self, x: int = 0, y: int = 0, infinity: bool = False):
        self.x = x
        self.y = y
        self.infinity = infinity

    def __eq__(self, other) -> bool:
        if not isinstance(other, CurvePoint):
            return False
        if self.infinity and other.infinity:
            return True
        if self.infinity != other.infinity:
            return False
        return self.x == other.x and self.y == other.y

    def __repr__(self) -> str:
        if self.infinity:
            return "CurvePoint(infinity)"
        return f"CurvePoint(x=0x{self.x:064x}, y=0x{self.y:064x})"

    def is_infinity(self) -> bool:
        return self.infinity

    def to_affine(self):
        return CurvePoint(self.x, self.y, self.infinity)


INFINITY = CurvePoint(0, 0, True)


def is_point_on_curve(point: CurvePoint, curve=Secp256r1) -> bool:
    if point.infinity:
        return True
    if point.x < 0 or point.x >= curve.p:
        return False
    if point.y < 0 or point.y >= curve.p:
        return False
    lhs = mod_mul(point.y, point.y, curve.p)
    rhs = mod_add(
        mod_add(
            mod_pow(point.x, 3, curve.p),
            mod_mul(curve.a, point.x, curve.p),
            curve.p,
        ),
        curve.b,
        curve.p,
    )
    return lhs == rhs


def validate_point(point: CurvePoint, curve=Secp256r1) -> None:
    if not isinstance(point, CurvePoint):
        raise TypeError("Point must be a CurvePoint instance")
    if point.infinity:
        raise ValueError("Point at infinity is not allowed")
    if not is_point_on_curve(point, curve):
        raise ValueError("Point is not on the curve")


def point_double(point: CurvePoint, curve=Secp256r1) -> CurvePoint:
    if point.infinity:
        return CurvePoint(0, 0, True)

    x1, y1 = point.x, point.y

    if y1 == 0:
        return CurvePoint(0, 0, True)

    numerator = mod_add(
        mod_mul(3, mod_mul(x1, x1, curve.p), curve.p),
        curve.a,
        curve.p,
    )
    denominator = mod_mul(2, y1, curve.p)
    lam = mod_mul(numerator, mod_inv(denominator, curve.p), curve.p)

    x3 = mod_sub(mod_sub(mod_mul(lam, lam, curve.p), x1, curve.p), x1, curve.p)
    y3 = mod_sub(mod_mul(lam, mod_sub(x1, x3, curve.p), curve.p), y1, curve.p)

    return CurvePoint(x3, y3, False)


def point_add(p1: CurvePoint, p2: CurvePoint, curve=Secp256r1) -> CurvePoint:
    if p1.infinity:
        return CurvePoint(p2.x, p2.y, p2.infinity)
    if p2.infinity:
        return CurvePoint(p1.x, p1.y, p1.infinity)

    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y

    if x1 == x2:
        if y1 != y2:
            return CurvePoint(0, 0, True)
        else:
            return point_double(p1, curve)

    lam = mod_mul(
        mod_sub(y2, y1, curve.p),
        mod_inv(mod_sub(x2, x1, curve.p), curve.p),
        curve.p,
    )

    x3 = mod_sub(mod_sub(mod_mul(lam, lam, curve.p), x1, curve.p), x2, curve.p)
    y3 = mod_sub(mod_mul(lam, mod_sub(x1, x3, curve.p), curve.p), y1, curve.p)

    return CurvePoint(x3, y3, False)


SCALAR_BITS = 256


def scalar_mult_constant_time(scalar: int, point: CurvePoint, curve=Secp256r1) -> CurvePoint:
    if point.infinity:
        return CurvePoint(0, 0, True)

    scalar = scalar % curve.n
    if scalar < 0:
        scalar += curve.n

    result = CurvePoint(0, 0, True)
    addend = CurvePoint(point.x, point.y, point.infinity)

    for i in range(SCALAR_BITS):
        bit = (scalar >> i) & 1

        tmp = point_add(result, addend, curve)
        new_x = constant_time_select(bit, tmp.x, result.x)
        new_y = constant_time_select(bit, tmp.y, result.y)
        new_inf = constant_time_select(bit, 1 if tmp.infinity else 0, 1 if result.infinity else 0)
        result.x, result.y, result.infinity = new_x, new_y, bool(new_inf)

        addend = point_double(addend, curve)

    return result


def scalar_mult(scalar: int, point: CurvePoint, curve=Secp256r1) -> CurvePoint:
    return scalar_mult_constant_time(scalar, point, curve)


def scalar_mult_base(scalar: int, curve=Secp256r1) -> CurvePoint:
    base_point = CurvePoint(curve.Gx, curve.Gy, False)
    return scalar_mult_constant_time(scalar, base_point, curve)
