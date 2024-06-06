import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parents[1]))
from paillier.src.paillier import generate_paillier_keypair
np.set_printoptions(precision=18)

p, q = generate_paillier_keypair()
data = np.random.random((2, 2))
print(data)
encry = p.encrypt(data)
print(q.decrypt(encry))