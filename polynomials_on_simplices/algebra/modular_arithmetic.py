"""Functionality for modular arithmetic."""


class IntegerModuloN:
    r"""
    Integer modulo n (element of :math:`\mathbb{Z}/n\mathbb{Z})`.

    We have

    .. math:: \mathbb{Z}/n\mathbb{Z} = \mathbb{Z}/\sim,

    where

    .. math:: a \sim b \text{ if } a \bmod n = b \bmod n \iff \exists c \in n\mathbb{Z} \text{ such that } a + c = b.

    This class defines the ring structure of integers modulo n.

    **Addition:**

    .. math:: + : \mathbb{Z}/n\mathbb{Z} \times \mathbb{Z}/n\mathbb{Z} \to \mathbb{Z}/n\mathbb{Z},

    .. math:: [a] + [b] = [a + b].

    **Multiplication:**

    .. math:: \cdot : \mathbb{Z}/n\mathbb{Z} \times \mathbb{Z}/n\mathbb{Z} \to \mathbb{Z}/n\mathbb{Z},

    .. math:: [a] \cdot [b] = [a \cdot b].
    """

    def __init__(self, value, n):
        """
        :param value: Value of the integer.
        :param n: Modulus of the integer.
        """
        self.i = value % n
        self.n = n

    def __repr__(self):
        return "polynomials_on_simplices.algebra.modular_arithmetic.IntegerModuloN(" + str(self.i) + ", " + str(self.n) + ")"

    def __str__(self):
        return str(self.i) + "_" + str(self.n)

    def __hash__(self):
        return self.i

    def __eq__(self, other):
        if isinstance(other, IntegerModuloN):
            if self.n != other.n:
                raise ValueError("Can not compare integer modulo " + str(self.n)
                                 + "with integer modulo " + str(other.n) + ".")
            return self.i == other.i
        return self.i == other

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        if isinstance(other, IntegerModuloN):
            if self.n != other.n:
                raise ValueError("Can not compare integer modulo " + str(self.n)
                                 + "with integer modulo " + str(other.n) + ".")
            return self.i < other.i
        return self.i < other

    def __gt__(self, other):
        if isinstance(other, IntegerModuloN):
            if self.n != other.n:
                raise ValueError("Can not compare integer modulo " + str(self.n)
                                 + "with integer modulo " + str(other.n) + ".")
            return self.i > other.i
        return self.i > other.i

    def __le__(self, other):
        return not (self > other)

    def __ge__(self, other):
        return not (self < other)

    def __neg__(self):
        return IntegerModuloN(-self.i, self.n)

    def __add__(self, other):
        if isinstance(other, IntegerModuloN):
            if self.n != other.n:
                raise ValueError("Can not add integer modulo " + str(self.n)
                                 + "with integer modulo " + str(other.n) + ".")
            return IntegerModuloN(self.i + other.i, self.n)
        return IntegerModuloN(self.i + other, self.n)

    def __sub__(self, other):
        if isinstance(other, IntegerModuloN):
            if self.n != other.n:
                raise ValueError("Can not subtract integer modulo " + str(other.n)
                                 + "from integer modulo " + str(self.n) + ".")
            return IntegerModuloN(self.i - other.i, self.n)
        return IntegerModuloN(self.i - other, self.n)

    def __mul__(self, other):
        if isinstance(other, IntegerModuloN):
            if self.n != other.n:
                raise ValueError("Can not multiply integer modulo " + str(self.n)
                                 + "with integer modulo " + str(other.n) + ".")
            return IntegerModuloN(self.i * other.i, self.n)
        return IntegerModuloN(self.i * other, self.n)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return -(self - other)

    def __rmul__(self, other):
        return self * other

    def __int__(self):
        return int(self.i)
