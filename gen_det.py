#!/usr/bin/python3
import os
import cv2
import json
from glob import glob
import numpy as np
import argparse
from matplotlib import pyplot as plt
# python gen_det.py --img_path val\img\ --det_path val\val_masks\
# python eval.py --img_path val\img\ --gt_path val\annotation\ --det_path val\val_masks\
# accuracy = 0.7964628925168669

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Evaluation Script for segmentation of Gall Bladder Images')
	parser.add_argument('-i', '--img_path', type=str, default='img', required=True, help="Path for the image folder")
	parser.add_argument('-d', '--det_path', type=str, default='det', required=True, help="Path for the detected masks folder")

	args = parser.parse_args()
	img_files = sorted(glob(os.path.join(args.img_path, "*jpg")))
	# det_files = sorted(glob(os.path.join(args.det_path, "*")))
	n = len(img_files)

	for i in range(0,n):
		# read the images
		img = cv2.imread(img_files[i], 0)

		# pre-processing the image to remove noise/smoothen
		
		filter = np.array([[1/16,1/8,1/16], [1/8,1/4,1/8], [1/16,1/8,1/16]])
		smoothen_img_1=cv2.filter2D(img,-1,filter)
		for k in range(0,2):
			filter = np.array([[1/16,1/8,1/16], [1/8,1/4,1/8], [1/16,1/8,1/16]])
			smoothen_img_1=cv2.filter2D(smoothen_img_1,-1,filter)
		filter = np.array([[0, -1, 0], [-1, 5, 1], [0, -1, 0]])
		smoothen_img_1=cv2.filter2D(smoothen_img_1,-1,filter)
		# smoothen_img_1=cv2.filter2D(smoothen_img_1,-1,filter)
		img = smoothen_img_1
		# cv2.imwrite(args.det_path+os.path.basename(img_files[i]), img)

		# img =  cv2.GaussianBlur(img, (11, 11), 0)
		# img = cv2.medianBlur(img,3)

		avg = np.mean(img)

		# taking different thresholds for the image for superposition in later stages
		ret, thresh1 = cv2.threshold(img, 0.8*avg, 255, cv2.THRESH_BINARY_INV)
		ret, thresh2 = cv2.threshold(img, 0.7*avg, 1.15*avg, cv2.THRESH_BINARY_INV)
		ret, thresh3 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)

		# post-processing of image after finding thresholds
		thresh1 = cv2.Laplacian(thresh1,cv2.CV_8U)
		thresh1 = cv2.Laplacian(thresh1,cv2.CV_8U)
		thresh1 =  cv2.GaussianBlur(thresh1, (11, 11), 0)

		thresh2 = cv2.Laplacian(thresh2,cv2.CV_8U)
		thresh2 = cv2.Laplacian(thresh2,cv2.CV_8U)
		thresh2 = cv2.GaussianBlur(thresh2, (11,11), 0)


		# finding contours for the post-processed images
		contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours1, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours2, hierarchy = cv2.findContours(thresh3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		# create a new empty image for generating masks
		temp = np.zeros(img.shape, np.uint8)

		hull = []
		# finding the convex hull for contours and creating masks based on it's area
		for j in range(len(contours)):
			hull = cv2.convexHull(contours[j])
			if 6000 < cv2.contourArea(hull) < 120000:
				cv2.drawContours(temp, [hull], -1, (255, 255, 255), -1)
		for j in range(len(contours1)):
			hull = cv2.convexHull(contours1[j])
			if 7000 < cv2.contourArea(hull) < 120000:
				cv2.drawContours(temp, [hull], -1, (255, 255, 255), -1)
		for j in range(len(contours2)):
			hull = cv2.convexHull(contours2[j])
			if 4000 < cv2.contourArea(hull) < 150000:
				cv2.drawContours(temp, [hull], -1, (255, 255, 255), 1)

		# create a padded image for removing parts that are not required
		temp2 = np.zeros(img.shape, np.uint8)
		tup = img.shape
		y1 = tup[0]/8
		x1 = tup[1]/8
		tours = np.array([ [x1,y1], [x1,7*y1], [7*x1, 7*y1], [7*x1,y1] ], dtype = np.int32)
		cv2.fillPoly(temp2, pts =[tours], color=(255,255,255))


		temp = cv2.bitwise_and(temp,temp,mask = temp2)

		cv2.imwrite(args.det_path+os.path.basename(img_files[i]), temp)
