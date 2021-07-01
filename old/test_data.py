import numpy as np
import random

def main():
    delta = 1
    drag = 2
    epsilon = 0.1

    X = np.array(([1], [5]))
    print(X)
    A = np.array(([1, delta], [0, 1 - drag]))

    N = 10
    for i in range(N):
        X = np.matmul(A,  X + np.array(([0], [delta * random.gauss(0, epsilon)])))
        print(X)

main()