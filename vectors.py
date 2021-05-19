import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def mine():
    ala = None
    if ala is None:
        print("hello")
    f = open("image/IMG_0000.pgm", 'rb')
    image = read_pgm2(f)
    f.close()

    size = (80, 60)

    x_pos = np.array([np.arange(size[0]), ]*size[1]).astype(int)
    y_pos = np.array([np.arange(size[1]), ]*size[0]).transpose().astype(int)

    dx = np.array([np.linspace(-5.0, 5.0, size[0]), ]*size[1]).astype(int)
    dy = (np.array([np.linspace(5.0, -5.0, size[1]), ]*size[0]).transpose()).astype(int)

    vec_len = np.sqrt(dx**2+dy**2)

    diff = np.amax(vec_len)-np.amin(vec_len)
    p = (vec_len-np.amin(vec_len))/diff*1.0

    x_pos = x_pos.reshape(x_pos.size)
    y_pos = y_pos.reshape(y_pos.size)

    dx = dx.reshape(dx.size)
    dy = dy.reshape(dy.size)


    plt.imshow(image)
    plt.quiver(x_pos, y_pos, dx, dy, p, scale=1, scale_units='xy')
    plt.show()

    height, width = image.shape[:2]
    res = cv.resize(image, (8 * width, 8 * height), interpolation=cv.INTER_CUBIC)

    for i in range(x_pos.size):
        cv.arrowedLine(res,(x_pos[i]*8,y_pos[i]*8),((x_pos[i]+dx[i])*8,(y_pos[i]-dy[i])*8),(0,0,0), 1)

    print(np.min(res))

    cv.imshow("img", res)
    cv.waitKey(0)




def read_pgm2(pgm_file):
    """Return raster of image from file P2 .pgm"""
    assert pgm_file.readline() != 'P2\n'

    (width, height) = [int(i) for i in pgm_file.readline().split()]
    max_value = int(pgm_file.readline())

    raster = []
    for _ in range(height):
        row = [int((int(i)/max_value)*255) for i in pgm_file.readline().split()]
        raster.append(row)

    return np.array(raster).astype('uint8')


if __name__ == '__main__':
    mine()