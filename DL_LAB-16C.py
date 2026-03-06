import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\shaik\Downloads\cat.image.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

_,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.subplot(121)
plt.imshow(gray,cmap='gray')
plt.title("Gray Image")

plt.subplot(122)
plt.imshow(thresh,cmap='gray')
plt.title("Threshold Image")

plt.show()
