import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import math
import torch


def normalize_the_target_plane_coordiantes(i, j):

    # basename = baseDir.split('/')[-1]+'_'
    # rgb = np.asarray(Image.open(os.path.join(baseDir, basename+'rgb.png'))) / 255.0

    # print(target_image.shape)
    
    # target_image[0].apply_(lambda i : 2*i/width - 1)
    # target_image[1].apply_(lambda j : 2*j/height - 1)

    # u = np.divide(np.dot(2, i), width) - 1
    # v = np.divide(np.dot(2, j), height) - 1

    u = 2 * i / width - 1
    v = 2 * j / height - 1
    # print(target_image.shape)

    return u, v


def calculate_spherical_coordinates(u,v):
    # lo = np.divide(np.dot((1-u),np.pi),2)
    # la = np.divide(np.dot((1-v),np.pi),2)

    # x = np.dot(np.sin(la), np.cos(lo))
    # y = np.cos(la)
    # z = np.dot(np.sin(la), np.sin(lo))

    lo = (1-u) * math.pi / 2
    la = (1-v) * math.pi / 2

    x = int(np.sin(la) * np.cos(lo))
    y = int(np.cos(la))
    z = int(np.sin(la) * np.sin(lo))

    return x, y, z

def calculate_radius_of_fisheye_image_R_and_coordinates_of_points_on_fisheye_image(x,y,z, width):

    # dist = np.divide(width,np.pi)
    # r = np.sqrt(np.dot(np.square(x),np.square(y)))
    # phi = np.arccos(z)
    # R = np.dot(dist,phi)

    # x_src = np.divide(np.dot(R,x), r)
    # y_src = np.divide(np.dot(R,y), r)

    dist = width / math.pi
    r = np.sqrt(np.square(x) * np.square(y))
    phi = np.arccos(z)
    Radi = dist * phi

    x_src = (Radi * x / r).astype(int)
    y_src = (Radi * y / r).astype(int)

    return x_src, y_src


if __name__ == '__main__':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    img = Image.open('./fisheyeimage.jpg')
    tf = transforms.ToTensor()
    img_t = tf(img)

    print(img_t.shape)

    width, height = img_t.shape[1:]
    print(width, height)

    img_t = img_t.permute(1,2,0)
    print(img_t.shape)
    print(width, height)
    
    target_image = torch.ones_like(img_t)

    for i in range(width):
        for j in range(height):
            for k in range(3):
                u, v = normalize_the_target_plane_coordiantes(i,j)

                x,y,z = calculate_spherical_coordinates(u,v)
                x_src, y_src = calculate_radius_of_fisheye_image_R_and_coordinates_of_points_on_fisheye_image(x,y,z,width)
                target_image[i,j,k] = img_t[int(x_src), int(y_src),k]
    # plt.imshow(target_image)
    plt.imshow(img_t)
    plt.show()