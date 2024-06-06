"""
    author: heran,
    This is a paillier algorithm based on numpy format,Support for single process and simple multi-process parallel;
    Maybe one day, it will be adapted to other distributed frameworks,but now ,make do with it,

"""


from util import getprimeover, invert, mulmod, powmod, isqrt
from encoding import EncodedNumber
from typing import Tuple
import random
from numpy.core import ndarray as Array
import numpy as np
import multiprocessing as mp
from functools import reduce


def dot_reduce(x):
    """
    concatenate blocks
    """
    return reduce(lambda item1, item2: np.append(item1, item2, axis=1), x)


def toArray(x):
    """
    just use for multi
    """
    return x.toArray()


def dot_block(x):
    """
    dot block
    """
    return np.dot(x[0], x[1].T)


def array_split(x, partitions):
    """
    x:array , will split
    partitions: int , the number of the matrix to be split
    """
    if partitions > x.shape[0]:
        partitions = x.shape[0]
    return np.array_split(x, partitions)


def array_trans_split(map_iter):
    """
    maybe not used
    """
    data = reduce(lambda x, y: np.append(x, y, axis=0), map_iter)
    return data


def pool_with_func(func, _iter, processes):
    """
    pool with process
    """
    pool = mp.Pool(processes=processes)
    r = pool.map(func, _iter)
    pool.close()
    pool.join()
    return r


def get_partitions_processes(processes, partitions):
    """
    get process num and data partitions
    default
    process = cpu src's num - 1,
    partitions = process * 4
    """
    processes = processes or mp.cpu_count() - 1
    partitions = partitions or processes * 4
    return processes, partitions


class PaillierPublicKey(object):
    """Contains a public key and associated encryption methods.

    Args:

      n (int): the modulus of the public key - see Paillier's paper.

    Attributes:
      g (int): part of the public key - see Paillier's paper.
      n (int): part of the public key - see Paillier's paper.
      nsquare (int): :attr:`n` ** 2, stored for frequent use.
      max_int (int): Maximum int that may safely be stored. This can be
        increased, if you are happy to redefine "safely" and lower the
        chance of detecting an integer overflow.
    """

    def __init__(self, n):
        self.g = n + 1
        self.n = n
        self.nsquare = n * n
        self.max_int = n // 3 - 1

    def __repr__(self):
        publicKeyHash = hex(hash(self))[2:]
        return "<PaillierPublicKey {}>".format(publicKeyHash[:10])

    def __eq__(self, other):
        return self.n == other

    def __hash__(self):
        return hash(self.n)

    def encrypt_(self, value, precision=None):
        encoding = EncodedNumber.encode(self.n, value, precision)

        ciphertext = self.raw_encrypt(encoding.encoding, r_value=1)

        encrypted_number = EncryptedNumber(self.n, ciphertext, encoding.exponent)
        encrypted_number.obfuscate()
        return encrypted_number

    def encrypt_exp(self, x):
        r = self.encrypt_(x)
        return np.array([r.ciphertext(False), r.exponent])

    def encrypt(self, value, precision=None, is_pool=False, processes=None, partitions=None):
        if is_pool:
            processes, partitions = get_partitions_processes(processes, partitions)
            data_iter = array_split(value, partitions=processes)
            value_iter = pool_with_func(self.encrypt_exp, data_iter, processes)
            r = reduce(lambda x, y: np.append(x, y, axis=1), value_iter)
            return EncryptedNumber(self.n, r[0, :, :], r[1, :, :])

        else:
            return self.encrypt_(value)

    def get_nude(self, plaintext):

        if self.n - self.max_int <= plaintext < self.n:
            # Very large plaintext, take a sneaky shortcut using inverses
            neg_plaintext = self.n - plaintext  # = abs(plaintext - nsquare)
            # avoid using gmpy2's mulmod when a * b < c
            neg_ciphertext = (self.n * neg_plaintext + 1) % self.nsquare
            # todo :may some problem
            nude_ciphertext = invert(neg_ciphertext, self.nsquare)
        else:
            # we chose g = n + 1, so that we can exploit the fact that
            # (n+1)^plaintext = n*plaintext + 1 mod n^2
            nude_ciphertext = (self.n * plaintext + 1) % self.nsquare

        return nude_ciphertext

    def raw_encrypt(self, plaintext, r_value=None):
        nude_ciphertext = np.frompyfunc(self.get_nude, 1, 1)(plaintext)
        r = r_value or self.get_random_lt_n()
        obfuscator = powmod(r, self.n, self.nsquare)
        # todo : may merge
        return np.frompyfunc(mulmod, 3, 1)(nude_ciphertext, obfuscator, self.nsquare)

    def get_random_lt_n(self):
        """Return a cryptographically random number less than :attr:`n`"""
        return random.SystemRandom().randrange(1, self.n)


class PaillierPrivateKey(object):
    def __init__(self, n, p, q):
        if not p * q == n:
            raise ValueError('given public key does not match the given p and q.')
        if p == q:
            # check that p and q are different, otherwise we can't compute p^-1 mod q
            raise ValueError('p and q have to be different')
        self.n = n
        if q < p:  # ensure that p < q.
            self.p = q
            self.q = p
        else:
            self.p = p
            self.q = q
        self.psquare = self.p * self.p

        self.qsquare = self.q * self.q
        self.p_inverse = invert(self.p, self.q)
        self.hp = self.h_function(self.p, self.psquare)
        self.hq = self.h_function(self.q, self.qsquare)

    @staticmethod
    def from_totient(public_key, totient):
        """given the totient, one can factorize the modulus

        The totient is defined as totient = (p - 1) * (q - 1),
        and the modulus is defined as modulus = p * q

        Args:
          public_key (PaillierPublicKey): The corresponding public
            key
          totient (int): the totient of the modulus

        Returns:
          the :class:`PaillierPrivateKey` that corresponds to the inputs

        Raises:
          ValueError: if the given totient is not the totient of the modulus
            of the given public key
        """
        p_plus_q = public_key.n - totient + 1
        p_minus_q = isqrt(p_plus_q * p_plus_q - public_key.n * 4)
        q = (p_plus_q - p_minus_q) // 2
        p = p_plus_q - q
        if not p * q == public_key.n:
            raise ValueError('given public key and totient do not match.')
        return PaillierPrivateKey(public_key, p, q)

    def __repr__(self):
        pub_repr = repr(self.n)
        return "<PaillierPrivateKey for {}>".format(pub_repr)

    def decrypt_(self, encrypted_number):
        encoded = self.decrypt_encoded(encrypted_number)
        return encoded.decode()

    def decrypt(self, value, is_pool=False, processes=None, partitions=None):
        if is_pool:
            processes, partitions = get_partitions_processes(processes, partitions)
            value = value.toArray()
            data_iter = array_split(value, partitions=processes)
            value_iter = pool_with_func(self.decrypt_, data_iter, processes)
            r = reduce(lambda x, y: np.append(x, y, axis=0), value_iter)
            return r

        else:
            return self.decrypt_(value)

    def decrypt_encoded(self, encrypted_number, Encoding=None):
        if Encoding is None:
            Encoding = EncodedNumber

        if isinstance(encrypted_number, EncryptedNumber):
            if self.n != encrypted_number.n:
                raise ValueError('encrypted_number was encrypted against a '
                                 'different key!')

            encoded = self.raw_decrypt(encrypted_number.ciphertext(be_secure=False))
            return Encoding(self.n, encoded, encrypted_number.exponent)

        if isinstance(encrypted_number, Array):
            exponent = np.frompyfunc(lambda x: x.exponent, 1, 1)(encrypted_number)
            encoded = np.frompyfunc(lambda x: self.raw_decrypt(x.ciphertext(be_secure=False)), 1, 1)(encrypted_number)
            return Encoding(self.n, encoded, exponent)

    def raw_decrypt(self, ciphertext):

        def crt_func(x):
            mq = mulmod(self.l_function(powmod(x, self.q - 1, self.qsquare),
                                        self.q),
                        self.hq, self.q)

            mp = mulmod(self.l_function(powmod(x, self.p - 1, self.psquare),
                                        self.p),
                        self.hp, self.p)
            u = mulmod(mq - mp, self.p_inverse, self.q)
            return mp + (u * self.p)

        return np.frompyfunc(crt_func, 1, 1)(ciphertext)

    def h_function(self, x, xsquare):
        """Computes the h-function as defined in Paillier's paper page 12,
        'Decryption using Chinese-remaindering'.
        """
        return invert(self.l_function(powmod(self.n + 1, x - 1, xsquare), x), x)

    def l_function(self, x, p):
        """Computes the L function as defined in Paillier's paper. That is: L(x,p) = (x-1)/p"""
        import gmpy2
        s = gmpy2.floor_div(x - 1, p)
        return s
        # return (x - 1) // p

    def __eq__(self, other):
        return self.p == other.p and self.q == other.q

    def __hash__(self):
        return hash((self.p, self.q))


class EncryptedNumber(object):
    def __init__(self, n, ciphertext, exponent):
        self.n = n
        self.nsquare = n * n
        self.max_int = n // 3 - 1
        self.__ciphertext = ciphertext
        self.exponent = exponent
        self.__is_obfuscated = False

    def __add__(self, other):
        """Add an int, float, `EncryptedNumber` or `EncodedNumber`."""
        if isinstance(other, EncryptedNumber):
            return self._add_encrypted(other)
        elif isinstance(other, EncodedNumber):
            return self._add_encoded(other)
        else:
            return self._add_scalar(other)

    def __radd__(self, other):
        """Called when Python evaluates `34 + <EncryptedNumber>`
        Required for builtin `sum` to work.
        """
        return self.__add__(other)

    def __mul__(self, other):
        """Multiply by an int, float, or EncodedNumber."""
        if isinstance(other, EncryptedNumber):
            raise NotImplementedError('Good luck with that...')

        if isinstance(other, EncodedNumber):
            encoding = other
        else:
            encoding = EncodedNumber.encode(self.n, other)
        product = self._raw_mul(encoding.encoding)
        exponent = self.exponent + encoding.exponent
        return EncryptedNumber(self.n, product, exponent)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return other + (self * -1)

    def __truediv__(self, scalar):
        return self.__mul__(1 / scalar)

    def ciphertext(self, be_secure=True):
        if be_secure and not self.__is_obfuscated:
            self.obfuscate()
        return self.__ciphertext

    def decrease_exponent_to(self, new_exp):
        if np.any(new_exp > self.exponent):
            raise ValueError('New exponent %i should be more negative than '
                             'old exponent %i' % (new_exp, self.exponent))
        # new_base = np.power(EncodedNumber.BASE, self.exponent - new_exp)
        new_base = np.frompyfunc(lambda x: pow(EncodedNumber.BASE, x), 1, 1)(self.exponent - new_exp)
        multiplied = self * new_base
        multiplied.exponent = new_exp

        return multiplied

    def get_random_lt_n(self):
        """Return a cryptographically random number less than :attr:`n`"""
        return random.SystemRandom().randrange(1, self.n)

    def obfuscate(self):
        r = self.get_random_lt_n()
        r_pow_n = powmod(r, self.n, self.nsquare)
        self.__ciphertext = np.frompyfunc(mulmod, 3, 1)(self.__ciphertext, r_pow_n, self.nsquare)
        self.__is_obfuscated = True

    def _add_encrypted(self, other):
        """Returns E(a + b) given E(a) and E(b).

        Args:
          other (EncryptedNumber): an `EncryptedNumber` to add to self.

        Returns:
          EncryptedNumber: E(a + b), calculated by taking the product
            of E(a) and E(b) modulo :attr:`~PaillierPublicKey.n` ** 2.

        Raises:
          ValueError: if numbers were encrypted against different keys.
        """
        if self.n != other.n:
            raise ValueError("Attempted to add numbers encrypted against "
                             "different public keys!")

        # In order to add two numbers, their exponents must match.
        a, b = self, other
        if isinstance(a.exponent, Array):
            mins = np.minimum(a.exponent, b.exponent)

            if np.any(a.exponent != mins):
                a = self.decrease_exponent_to(mins)

            if np.any(b.exponent != mins):
                b = b.decrease_exponent_to(mins)

        else:
            if a.exponent > b.exponent:
                a = self.decrease_exponent_to(b.exponent)
            elif a.exponent < b.exponent:
                b = b.decrease_exponent_to(a.exponent)

        sum_ciphertext = a._raw_add(a.ciphertext(False), b.ciphertext(False))
        return EncryptedNumber(a.n, sum_ciphertext, a.exponent)

    def _add_scalar(self, scalar):
        encoded = EncodedNumber.encode(self.n, scalar,
                                       max_exponent=self.exponent)
        return self._add_encoded(encoded)

    def _add_encoded(self, encoded):
        """Returns E(a + b), given self=E(a) and b.

        Args:
          encoded (EncodedNumber): an :class:`EncodedNumber` to be added
            to `self`.

        Returns:
          EncryptedNumber: E(a + b), calculated by encrypting b and
            taking the product of E(a) and E(b) modulo
            :attr:`~PaillierPublicKey.n` ** 2.

        Raises:
          ValueError: if scalar is out of range or precision.
        """
        if self.n != encoded.n:
            raise ValueError("Attempted to add numbers encoded against "
                             "different public keys!")

        # In order to add two numbers, their exponents must match.
        a, b = self, encoded

        if isinstance(a.exponent, Array):
            mins = np.minimum(a.exponent, b.exponent)
            if np.any(a.exponent != mins):
                a = self.decrease_exponent_to(mins)

            if np.any(b.exponent != mins):
                b = b.decrease_exponent_to(mins)
        else:
            if a.exponent > b.exponent:
                a = self.decrease_exponent_to(b.exponent)
            elif a.exponent < b.exponent:
                b = b.decrease_exponent_to(a.exponent)
        # Don't bother to salt/obfuscate in a basic operation, do it
        # just before leaving the computer.
        encrypted_scalar = PaillierPublicKey(a.n).raw_encrypt(b.encoding, 1)
        sum_ciphertext = a._raw_add(a.ciphertext(False), encrypted_scalar)
        return EncryptedNumber(a.n, sum_ciphertext, a.exponent)

    def _raw_add(self, e_a, e_b):
        """Returns the integer E(a + b) given ints E(a) and E(b).

        N.B. this returns an int, not an `EncryptedNumber`, and ignores
        :attr:`ciphertext`

        Args:
          e_a (int): E(a), first term
          e_b (int): E(b), second term

        Returns:
          int: E(a + b), calculated by taking the product of E(a) and
            E(b) modulo :attr:`~PaillierPublicKey.n` ** 2.
        """
        return np.frompyfunc(lambda x, y: mulmod(x, y, self.nsquare), 2, 1)(e_a, e_b)

    def _raw_mul(self, plaintext):
        """Returns the integer E(a * plaintext), where E(a) = ciphertext

        Args:
          plaintext (int): number by which to multiply the
            `EncryptedNumber`. *plaintext* is typically an encoding.
            0 <= *plaintext* < :attr:`~PaillierPublicKey.n`

        Returns:
          int: Encryption of the product of `self` and the scalar
            encoded in *plaintext*.

        Raises:
          TypeError: if *plaintext* is not an int.
          ValueError: if *plaintext* is not between 0 and
            :attr:`PaillierPublicKey.n`.
        """
        if not (isinstance(plaintext, int) or isinstance(plaintext, np.ndarray)):
            raise TypeError('Expected ciphertext to be int, not %s' %
                            type(plaintext))

        if np.any(plaintext < 0) or np.any(plaintext >= self.n):
            raise ValueError('Scalar out of bounds: %i' % plaintext)
        if np.any(self.n - self.max_int <= plaintext):
            # Very large plaintext, play a sneaky trick using inverses
            neg_c = np.frompyfunc(invert, 2, 1)(self.ciphertext(False), self.nsquare)
            neg_scalar = self.n - plaintext
            return np.frompyfunc(powmod, 3, 1)(neg_c, neg_scalar, self.nsquare)
        else:
            return np.frompyfunc(powmod, 3, 1)(self.ciphertext(False), plaintext, self.nsquare)

    def toArray(self):
        return np.frompyfunc(EncryptedNumber, 3, 1)(self.n, self.__ciphertext, self.exponent)

    @property
    def shape(self):
        if isinstance(self.exponent, np.ndarray):
            return self.exponent.shape
        else:
            raise TypeError("no shape！maybe (1,1) ,you can use np.ndarray")

    def dot(self, other, is_pool=False, processes=None, partitions=None):
        """
        The ciphertext matrix is multiplied by the plaintext matrix

              input: matrix_B:np.ndarray
              return EncryptedNumber()
              example:
                  Z[A] = EncryptedNumber()
                  A = np.array([[1,2,3],
                               [1,2,1]])

                  B = np.array([[1,1],
                               [1,2],
                               [0,3]])
                  Z[C]=Z[A].dot(B)=[[3,14],
                                  [3,8]]
        :param: other:np.ndarray
        :return: EncryptedNumber()
        """
        if isinstance(other, EncryptedNumber):
            raise ValueError("Do you believe in light?")

        if is_pool:
            return self.dot_(other, processes=processes, partitions=partitions)
        else:
            data = self.toArray()
            return toEncryptedNumber(np.dot(data, other))

    def dot_(self, other, processes=None, partitions=None):
        """
        args:
            nothing, If you see this code, thanks for filling in the comments for me.
        """
        processes, partitions = get_partitions_processes(processes, partitions)
        data_iter = array_split(self.toArray(), partitions=partitions)
        other_iter = array_split(other.T, partitions=partitions)
        other_iter_repeat = other_iter * len(data_iter)
        data_iter_repeat = reduce(lambda x, y: x + y, map(lambda x: [x] * len(other_iter), data_iter))
        r = zip(data_iter_repeat, other_iter_repeat)
        value_iter = np.array(pool_with_func(dot_block, r, processes))

        value_set = array_split(value_iter, len(data_iter))

        value = reduce(lambda item1, item2: np.append(item1, item2, axis=0),
                       pool_with_func(dot_reduce, value_set, processes))
        return value


def generate_paillier_keypair(n_length=1024) -> Tuple[PaillierPublicKey, PaillierPrivateKey]:
    """Return a new :class:`PaillierPublicKey` and :class:`PaillierPrivateKey`.

    Add the private key to *private_keyring* if given.

    Args:
      n_length: key size in bits.

    Returns:
      tuple: The generated :class:`PaillierPublicKey` and
      :class:`PaillierPrivateKey`
    """
    p = q = n = None
    n_len = 0
    while n_len != n_length:
        p = getprimeover(n_length // 2)
        q = p
        while q == p:
            q = getprimeover(n_length // 2)
        n = p * q
        n_len = n.bit_length()

    public_key = PaillierPublicKey(n)
    private_key = PaillierPrivateKey(n, p, q)

    return public_key, private_key


def toEncryptedNumber(arr):
    """
        an array for EncryptedNumber, trans to a EncryptedNumber object
    """
    data = np.frompyfunc(lambda x: (x.ciphertext(False), x.exponent), 1, 2)(arr)
    n = arr.flatten()[0].n
    return EncryptedNumber(n, data[0], data[1])
