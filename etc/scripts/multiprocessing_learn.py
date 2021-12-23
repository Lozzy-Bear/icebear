import time
import multiprocessing as mp
import numpy as np

# Note: Windows requires mp inside the __main__.

def counter1(num):
    cnt = 0
    for _ in range(num):
        cnt += 1
    print("counter 1 done")
    return cnt


def counter2(num):
    cnt = 0
    for _ in range(0, num, 2):
        cnt += 1
    print("counter 2 done")


if __name__ == '__main__':
    N = 2 * 10**3
    st = time.time()

    num_arr = np.arange(N)
    num_proc = 10

    with mp.Pool(processes=num_proc) as pool:
        results = pool.map(counter1, num_arr)
    pool.close()
    print(results)

    en = time.time()
    print(f'time: {en-st}')




