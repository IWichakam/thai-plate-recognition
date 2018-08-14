
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
import pytesseract as tess


def preprocess(img):
	cv2.imshow("Input",img)
	blur = cv2.GaussianBlur(img, (5,5), 0)
	gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

	sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
	cv2.imshow("Sobel", sobelx)

	ret, thresh = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imshow("Threshold", thresh)

	return thresh


def cleanPlate(plate):
	print("CLEANING PLATE. . .")
	gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

	_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
	imt, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	if contours:
		areas = [cv2.contourArea(c) for c in contours]
		max_index = np.argmax(areas)
		max_cnt = contours[max_index]
		max_cntArea = areas[max_index]

		x,y,w,h = cv2.boundingRect(max_cnt)

		if not ratioCheck(max_cntArea, w, h):
			return plate, None

		cleaned_final = thresh[y:y+h, x:x+w]

		return cleaned_final, [x,y,w,h]

	else:
		return plate, None


def extract_contours(thresh):
	element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(15, 5))
	morph = thresh.copy()
	cv2.morphologyEx(src=thresh, op=cv2.MORPH_CLOSE, kernel=element, dst=morph)
	cv2.imshow("Morphed", morph)
	#cv2.waitKey(0)

	imt, contours, hierarchy = cv2.findContours(morph, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
	
	return contours


def ratioCheck(area, width, height):
	ratio = float(width) / float(height)

	aspect = 4.7
	min = 1*aspect*1  # minimum area
	max = 250*aspect*250  # maximum area

	rmin = 2
	rmax = 5

	if (area < min or area > max) or (ratio < rmin or ratio > rmax):
		return False
	return True


def putText(img, x,y, text):
	fontpath = "font/THSarabunNew.ttf"
	font = ImageFont.truetype(fontpath, 48)
	img_pil = Image.fromarray(img)
	draw = ImageDraw.Draw(img_pil)
	draw.text((x, y),  text, font = font, fill=(0,255,0))
	img = np.array(img_pil)

	return img


def cleanAndRead(img, contours):
	for i,cnt in enumerate(contours):
		min_rect = cv2.minAreaRect(cnt)
		x,y,w,h = cv2.boundingRect(cnt)

		plate_img = img[y:y+h,x:x+w]

		clean_plate, rect = cleanPlate(plate_img)

		if rect:
			xt,yt,wt,ht = rect
			x,y,w,h = x+xt, y+yt, wt, ht
			cv2.imshow("Cleaned Plate", clean_plate)
			plate_im = Image.fromarray(clean_plate)
			text = tess.image_to_string(plate_im, 'tha')
			print("Detected Text : ", text)
			img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
			img = putText(img, x,y-h, text)
			cv2.imshow("Detected Plate", img)
			cv2.waitKey(0)


if __name__ == '__main__':
	print("DETECTING PLATE . . .")

	img = cv2.imread("img/thai2.jpg")
	img = cv2.resize(img, (512,256))

	thresh = preprocess(img)
	contours = extract_contours(thresh)

	cleanAndRead(img, contours)
