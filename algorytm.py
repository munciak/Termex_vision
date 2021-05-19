import numpy as np


def main():
    a = np.random.randint(127, size=(60, 80))
    b = np.random.randint(127, size=(60, 80))
    c = np.random.randint(127, size=(60, 80))
    z = np.concatenate([a,b,c])
    print(z.size)


if __name__ == '__main__':
    main()