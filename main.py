import torch
import PIL
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

img = PIL.Image.open('./fisheyeimage.jpg')
tf = transforms.ToTensor()
img_t = tf(img)

print(img_t.size())

img_t = img_t.permute(1,2,0)

print(img_t.size())

plt.imshow(img_t)
plt.show()