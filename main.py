import cv2
import matplotlib.pyplot as plt

from ImgCropper import ImgCropper


image = cv2.imread('images/nb.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cropped_img = ImgCropper().crop_image(image)

plt.axis("off")
plt.title("Cropped image")
plt.imshow(cropped_img)