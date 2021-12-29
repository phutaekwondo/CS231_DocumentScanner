import cv2 
import cs231
import numpy as np


path = "monhocRun/monhoc.jpg"

DEFAULT_THRES1 = 70
DEFAULT_THRES2 = 70
DEFAULT_BLUR = 5
DEFAULT_THICK = 4

PROCESS_HEIGHT = 700

img = cv2.imread( path,0 )
img_color = cv2.imread( path )
img = cs231.resizeImagebyH( img, 700)
img_color = cs231.resizeImagebyH( img_color, 700)

#blur
blur = DEFAULT_BLUR*2+1
img = cv2.GaussianBlur( img, (blur,blur), cv2.BORDER_DEFAULT )

# Canny
img = cv2.Canny( img, DEFAULT_THRES1, DEFAULT_THRES2 )

# dilate an erode
kernel = np.ones((5,5), np.uint8)
img = cv2.dilate(img, kernel)

img = cv2.erode(img, kernel)


#contour
contours, _ = cv2.findContours( img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

img_Contours = np.zeros( img_color.shape, np.uint8 )
img_Appro = np.zeros( img_color.shape, np.uint8 )
img_Contour_Appro = np.zeros( img_color.shape, np.uint8 )
colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
gray_value = 100
#for i in [5]:
for i in range(len(contours)):
	cv2.drawContours( img_Contours, [contours[i]], -1, colors[i%len(colors)], DEFAULT_THICK)
	cv2.drawContours( img_Contour_Appro, [contours[i]], -1, (gray_value,gray_value,gray_value), DEFAULT_THICK)

#appro

#for i in [5]:
for i in range(len(contours)):
	this_contour = contours[i]
	appro = cv2.approxPolyDP( this_contour, 0.01*cv2.arcLength( this_contour, True), True)

	for a in appro:
		for point in a :
			img_Appro = cv2.circle( img_Appro, point, 0, colors[i%len(colors)], 10 )
			img_Contour_Appro = cv2.circle( img_Contour_Appro , point, 0, colors[i%len(colors)], 10 )

cv2.imshow( "asdf", img_Contour_Appro)
cv2.waitKey(0)
cv2.imwrite("contour.jpg", img_Contours)
cv2.imwrite("contour_appro.jpg", img_Contour_Appro)
cv2.imwrite("appro.jpg", img_Appro )