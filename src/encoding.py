import math
import sys
import numpy as np
from numpy.core import ndarray as Array


class EncodedNumber(object):
    BASE = 16
    LOG2_BASE = math.log(BASE, 2)
    FLOAT_MANTISSA_BITS = sys.float_info.mant_dig

    def __init__(self, n, encoding: Array, exponent: Array):
        self.n = n
        self.max_int = n // 3 - 1
        self.encoding = encoding
        self.exponent = exponent

    @classmethod
    def encode(cls, n, scalar, precision=None, max_exponent=None):
        """Return an encoding of an int or float.

        This encoding is carefully chosen so that it supports the same
        operations as the Paillier cryptosystem.

        If *scalar* is a float, first approximate it as an int, `int_rep`:

            scalar = int_rep * (:attr:`BASE` ** :attr:`exponent`),

        for some (typically negative) integer exponent, which can be
        tuned using *precision* and *max_exponent*. Specifically,
        :attr:`exponent` is chosen to be equal to or less than
        *max_exponent*, and such that the number *precision* is not
        rounded to zero.

        Having found an integer representation for the float (or having
        been given an int `scalar`), we then represent this integer as
        a non-negative integer < :attr:`~PaillierPublicKey.n`.

        Paillier homomorphic arithemetic works modulo
        :attr:`~PaillierPublicKey.n`. We take the convention that a
        number x < n/3 is positive, and that a number x > 2n/3 is
        negative. The range n/3 < x < 2n/3 allows for overflow
        detection.

        Args:
          public_key (PaillierPublicKey): public key for which to encode
            (this is necessary because :attr:`~PaillierPublicKey.n`
            varies).
          scalar: an int or float to be encrypted.
            If int, it must satisfy abs(*value*) <
            :attr:`~PaillierPublicKey.n`/3.
            If float, it must satisfy abs(*value* / *precision*) <<
            :attr:`~PaillierPublicKey.n`/3
            (i.e. if a float is near the limit then detectable
            overflow may still occur)
          precision (float): Choose exponent (i.e. fix the precision) so
            that this number is distinguishable from zero. If `scalar`
            is a float, then this is set so that minimal precision is
            lost. Lower precision leads to smaller encodings, which
            might yield faster computation.
          max_exponent (int): Ensure that the exponent of the returned
            `EncryptedNumber` is at most this.

        Returns:
          EncodedNumber: Encoded form of *scalar*, ready for encryption
          against *public_key*.
        """
        # Calculate the maximum exponent for desired precision
        if isinstance(scalar, int) or isinstance(scalar, float):
            scalar = np.array(scalar)

        if np.issubdtype(scalar.dtype, np.int16) or np.issubdtype(scalar.dtype, np.int32) \
                or np.issubdtype(scalar.dtype, np.int64):
            prec_exponent = np.zeros(scalar.shape, dtype=np.int32)

        elif np.issubdtype(scalar.dtype, np.float16) \
                or np.issubdtype(scalar.dtype, np.float32) or np.issubdtype(scalar.dtype, np.float64):
            bin_flt_exponent = np.frexp(scalar)[1]
            bin_lsb_exponent = bin_flt_exponent - cls.FLOAT_MANTISSA_BITS
            prec_exponent = np.floor(bin_lsb_exponent / cls.LOG2_BASE).astype(np.int32)

        elif np.issubdtype(scalar.dtype, object):
            prec_exponent = np.zeros(scalar.shape, dtype=np.int32)
        else:
            raise TypeError("Don't know the precision of type %s."
                            % type(scalar))

        if max_exponent is None:
            exponent = prec_exponent
        else:
            exponent = np.minimum(max_exponent, prec_exponent).min()

        def power(x, scalar, n):
            return int(scalar * pow(cls.BASE, x)) % n

        int_rep = np.frompyfunc(lambda x, y, z: power(x, y, z), 3, 1)(-exponent, scalar, n)
        return cls(n, int_rep, exponent)

    # def decode(self):
    #     """Decode plaintext and return the result.
    #
    #     Returns:
    #       an int or float: the decoded number. N.B. if the number
    #         returned is an integer, it will not be of type float.
    #
    #     Raises:
    #       OverflowError: if overflow is detected in the decrypted number.
    #     """
    #     #
    #     # if self.encoding.any() > self.max_int or self.encoding>= self.n:
    #     #     raise OverflowError('Overflow detected in decrypted number')
    #     mask_true = self.encoding >= self.n - self.max_int
    #     mantissa = self.encoding - self.n * mask_true
    #     mask_exponent_true = self.exponent < 0
    #     mask_exponent_false = self.exponent >= 0
    #     exponent = - self.exponent * mask_exponent_true + self.exponent * mask_exponent_false
    #     r = np.frompyfunc(lambda x: pow(self.BASE, x), 1, 1)(exponent)
    #     r_t = 1 / r
    #     return mantissa * (r * mask_exponent_false + r_t * mask_exponent_true)

    def get_mantissa(self,x,y,z):
        if x >= self.n:
            # Should be mod n
            raise ValueError('Attempted to decode corrupted number')
        elif x <= self.max_int:
            # Positive
            mantissa = x
        elif x >= self.n - self.max_int:
            # Negative
            mantissa = x - self.n
        else:
            raise OverflowError('Overflow detected in decrypted number')
        return mantissa * pow(y,int(z))

    def decode(self):
        """Decode plaintext and return the result.

        Returns:
          an int or float: the decoded number. N.B. if the number
            returned is an integer, it will not be of type float.

        Raises:
          OverflowError: if overflow is detected in the decrypted number.
        """
        mantissa = np.frompyfunc(self.get_mantissa, 3, 1)(self.encoding,self.BASE, self.exponent)
        return mantissa

    def decrease_exponent_to(self, new_exp):
        if np.any(new_exp > self.exponent):
            raise ValueError('New exponent should be more negative than'
                             'old exponent ')
        factor = np.power(self.BASE, self.exponent - new_exp)
        new_enc = self.encoding * factor % self.n
        return self.__class__(self.n, new_enc, new_exp)
