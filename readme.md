# Paillier-Numpy

## Introduction

​		This project implements the partially homomorphic Paillier algorithm and extends and optimizes encryption, decryption, matrix multiplication, and other operations on Numpy matrices, thereby enhancing speed.

​		Main reference projects：https://github.com/data61/python-paillier

​		If there is a need for distributed computing, [this project](https://github.com/Galloroc1/Icaro) contains a Paillier implementation that I rewrote using Dask. It can perform distributed encryption, decryption, and computation using Dask's distribute functionality. However, it is worth noting that this is merely an experimental project.

## Setup

​		use pip to setup:

```
pip install paillier-numpy
```

## How to use

### Encrypt and decrypt

```
import numpy as np
from paillier import generate_paillier_keypair

# create public key and private key
p,q = generate_paillier_keypair()

# create random float
data = np.random.random_sample((100, 100))

# encrypt
encry = p.encrypt(data)

# decrypt
decry = q.decrypt(encry)

```



### Ciphertext addition and subtraction

```
p, q = generate_paillier_keypair()
data = np.random.random((2, 2))
encry = p.encrypt(data)

# float
encry2 = encry + 3.1415926
print(q.decrypt(encry2) == data + 3.1415926)
# [[ True  True]
#  [ True  True]]

# int
encry3 = encry + 666
print(q.decrypt(encry3) == data + 666)
# [[ True  True]
#  [ True  True]]

# matrix float
data2 = np.random.random_sample((2, 2))
encry4 = encry + data2
print(q.decrypt(encry4) == data + data2)
# [[ True  True]
#  [ True  True]]

# matrix int
data2 = np.random.randint(-100,100,(2, 2))
encry5 = encry + data2
print(q.decrypt(encry5) == data + data2)
# [[ True  True]
#  [ True  True]]
```

### Ciphertext Multiplication

```
p, q = generate_paillier_keypair()
data = np.random.random((2, 2))
encry = p.encrypt(data)

# int
encry2 = encry * 2
print(q.decrypt(encry2)==data*2)
# [[ True  True]
#  [ True  True]]

# float
encry2 = encry * 3.141592
print(q.decrypt(encry2)==data*3.141592)
# [[ True  True]
#  [ True  True]]
```



### Dot

**Note:** When performing matrix multiplication, there may be slight errors due to precision issues in the 16th decimal place.

```
p, q = generate_paillier_keypair()
data = np.random.random((2, 2))
data2 = np.random.random_sample((2,3))
encry = p.encrypt(data)

# [A].dot(B)
encry2 = encry.dot(data2)
print(q.decrypt(encry2)==data.dot(data2))
# [[ True  True  True]
#  [ True  True  True]]

# B.dot([A])
# !!!!!!!!!!!!!!!!!!!!!!
# [A] must  be converted to Numpy.ndarray
encry3 = data2.T.dot(encry.toArray())
print(q.decrypt(encry3)==data2.T.dot(data))
# [[ True  True]
#  [ True  True]
#  [ True  True]]
```



### Multi-process parallelism

**Note:** The sample code is as follows: where `partitions` is the number of blocks the matrix is divided into for multiprocessing use.

```
p, q = generate_paillier_keypair()
data = np.random.random((1000, 100))
data2 = np.random.random_sample((1000,100))

encry = p.encrypt(data)
# time : 0:00:00.624028
decry = q.decrypt(encry)
# time : 0:01:06.076635

encry = p.encrypt(data, is_pool=True, partitions=10)
# time : 0:00:00.282844
decry = q.decrypt(encry, is_pool=True, partitions=10)
# time : 0:00:04.417449
```

### DOT by Multi-process parallelism

**Note:** Recommend using multi-process matrix multiplication

```
p, q = generate_paillier_keypair()
data = np.random.random((100, 10))
data2 = np.random.random_sample((10,100))

encry = p.encrypt(data)
encry2 = encry.dot(data2)
# time cost: 0:00:21.922967

encry2 = p.encrypt(data, is_pool=True, partitions=10)
encry3 = encry2.dot(data2,is_pool=True,partitions=10)
# time cost: 0:00:01.600504
```

