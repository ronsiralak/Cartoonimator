import cv2
from imutils import face_utils
import numpy as np
import dlib
import output_face
import output_eye
import output_nose
import output_mouth
from os import listdir
from os.path import isfile, join
from PIL import Image

import time
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
from sklearn.metrics import confusion_matrix
import dataset


def webcam():

    cap = cv2.VideoCapture(0)

    while (True):

        ret, frame = cap.read()
        #อ่านค่าจาก webcam

        ret, frame_no_line = cap.read()
        #อ่านค่าจาก webcam เหมือนกัน แต่ frame จะเอาไปใส่เส้น frame_no_line เอาไว้แสดงผล ถ้าแสดงผล frame มันจะมีเส้นติดมาด้วย

        frame = cv2.flip(frame, 1)
        frame_no_line = cv2.flip(frame_no_line, 1)
        #กลับด้านซ้าย-ขวา พารามิเตอร์ 1 หมายถึง กลับซ้าย-ขวา ถ้า 0 หมายถึง บน-ล่าง

        height, width = frame.shape[:2]
        #เก็บขนาดของหน้า webcam (list ตัวที่ 0 และ 1 ของ shape คือ ความสูง และความยาว)

        width_center = int(width/2)
        height_center = int(height/2)
        #ศูนย์กลางของภาพ

        face_width = int(112/640*width)#ความยาวจากศูนย์กลางหน้าจนถึงขอบหน้าด้านข้าง
        face_height = int(160/480*height)#ความยาวจากศูนย์กลางหน้าจนถึงคาง-หัว
        eye_range = int(46/640*width)#ความยาวจากศูนย์กลางหน้าจนถึงศูนย์กลางตา
        eye_width = int(22/640*width)#ความยาวจากศูนย์กลางตาจนถึงหัวตา-หางตา
        eye_height = int(10/480*height)#ความยาวจากศูนย์กลางตาจนถึงขอบตาบน-ล่าง
        #ค่าต่างๆที่ตั้งเอาไว้ เพื่อกำหนดขนาดของวงรี

        cv2.line(frame,(0,height_center),(width,height_center),(0,255,0))
        cv2.line(frame,(width_center,0),(width_center,height),(0,255,0))
        #วาดเส้นศูนย์กลางภาพ โดยฟังก์ชัน line(image,(x1,y1),(x2,y2),color)

        cv2.ellipse(frame,(width_center,height_center),(face_width, face_height),0, 0, 360,(0, 255, 0 ), 1, 8 )
        #วาดวงรีใบหน้า โดยฟังก์ชัน ellipse(image,ศูนย์กลางของวงรี,ขนาดแกนเอกและแกนโทของวงรี,ความเอียงของวงรี,องศาเริ่มต้น,องศาสิ้นสุด ถ้าเป็น 0 ถึง 360 ก็คือ ครบวงพอดี, color, thickness, ประเภทของเส้น 8-connected line)

        cv2.ellipse(frame,(width_center + eye_range,height_center), (eye_width, eye_height), 0, 0, 360, (0, 255, 0), 1, 8)
        cv2.ellipse(frame,(width_center - eye_range,height_center), (eye_width, eye_height), 0, 0, 360, (0, 255, 0), 1, 8)
        #วาดวงรีตาทั้งสองข้าง

        cv2.imshow('put your face in the oval outline', frame)
        #ขึ้นหน้าต่าง webcam

        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite('captured_image.jpg', frame_no_line) #ถ่ายเป็นภาพที่ไม่มีเส้นแทน
            break
        #กด c (capture) เพื่อถ่าย

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #กด q (quit) เพื่อยกเลิก


    cap.release()
    cv2.destroyAllWindows()
    #ปิด webcam

def facial_landmarks():
    # initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor

    image = cv2.imread('captured_image.jpg')

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # load the input image, resize it, and convert it to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # get image size
    height, width, channels = image.shape

    # create blank image
    blank = np.zeros([height, width, 3], dtype=np.uint8)
    blank[:] = 255
    y_point = []
    x_point = []

    # detect faces in the grayscale image
    rects = detector(gray, 1)

	# loop over the face detections
    for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
        for (x, y) in shape:
			#cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
            cv2.circle(blank, (x, y), 1, (255, 0, 0), -1)
            y_point.append(y)
            x_point.append(x)

    top = max(y_point)
    bottom = min(y_point)
    left = min(x_point)
    right = max(x_point)

    blank_crop = blank[bottom:top,left:right]
    face_crop = image[bottom:top,left:right]
    height_face, width_face, channels = face_crop.shape
    eye_crop = face_crop[int(0.9/11*height_face):int(3/11*height_face),int(1.5/12*width_face):int(5/12*width_face)]
    eye_crop_blank = blank_crop[int(0.9/11*height_face):int(3/11*height_face),int(1.5/12*width_face):int(5/12*width_face)]
    nose_crop = face_crop[int(3/11*height_face):int(6.5/11*height_face),int(4/12*width_face):int(8/12*width_face)]
    nose_crop_blank = blank_crop[int(3/11*height_face):int(6.5/11*height_face),int(4/12*width_face):int(8/12*width_face)]
    mouth_crop = face_crop[int(6.3/11*height_face):int(9/11*height_face),int(3/12*width_face):int(9/12*width_face)]
    mouth_crop_blank = blank_crop[int(6.3/11*height_face):int(9/11*height_face),int(3/12*width_face):int(9/12*width_face)]
	# show the output image
    cv2.imwrite('input/face/face.jpg', face_crop)
    cv2.imwrite('input/face_dots/face_dots.jpg', blank_crop)
    cv2.imwrite('input/eye/eye.jpg', eye_crop)
    cv2.imwrite('input/eye_dots/eye_dots.jpg', eye_crop_blank)
    cv2.imwrite('input/nose/nose.jpg', nose_crop)
    cv2.imwrite('input/nose_dots/nose_dots.jpg', nose_crop_blank)
    cv2.imwrite('input/mouth/mouth.jpg', mouth_crop)
    cv2.imwrite('input/mouth_dots/mouth_dots.jpg', mouth_crop_blank)

    cv2.waitKey(0)

def overlay(gender):
    from PIL import Image
    img = cv2.imread("input/face_dots/face_dots.jpg")
    photo = Image.open('input/face_dots/face_dots.jpg')
    photo = photo.convert('RGB')

    width = photo.size[0]  # define W and H
    height = photo.size[1]

    # --------------------------------------------------------------------------------------------------------

    point = []
    x_point = []
    y_point = []
    c = 1

    for x in range(int(1.5 * width / 12), int(5 * width / 12)):
        for y in range(int(0.9 * height / 11), int(3 * height / 11)):
            RGB = photo.getpixel((x, y))
            R, G, B = RGB
            if R < 150:
                point.append((x, y))  # get point
                x_point.append(x)
                y_point.append(y)
                if c == 1:
                    x_eye = x
                    y_eye = y
                    c = c + 1
                    # get color and point of blue pixel from bottom to top

    left = min(x_point)
    right = max(x_point)
    top = max(y_point)
    bottom = min(y_point)

    eye = img[bottom:top, left:right]

    # ----------------------------------------------------------------------------------

    point = []
    x_point = []
    y_point = []
    c = 1

    for x in range(int(4 * width / 12), int(8 * width / 12)):
        for y in range(int(3 * height / 11), int(6.5 * height / 11)):
            RGB = photo.getpixel((x, y))
            R, G, B = RGB
            if R < 150:
                point.append((x, y))  # get point
                x_point.append(x)
                y_point.append(y)
                if c == 1:
                    x_nose = x
                    y_nose = y
                    c = c + 1
                    # get color and point of blue pixel from bottom to top

    left = min(x_point)
    right = max(x_point)
    top = max(y_point)
    bottom = min(y_point)

    nose = img[bottom:top, left:right]

    # ----------------------------------------------------------------------------------

    point = []
    x_point = []
    y_point = []
    c = 1

    for x in range(int(3 * width / 12), int(9 * width / 12)):
        for y in range(int(6.3 * height / 11), int(9 * height / 11)):
            RGB = photo.getpixel((x, y))
            R, G, B = RGB
            if R < 150:
                point.append((x, y))  # get point
                x_point.append(x)
                y_point.append(y)
                if c == 1:
                    x_mouth = x
                    y_mouth = y
                    c = c + 1
                    # get color and point of blue pixel from bottom to top

    left = min(x_point)
    right = max(x_point)
    top = max(y_point)
    bottom = min(y_point)

    mouth = img[bottom:top, left:right]

    # -----------------------------------------------------------------------------------------------------------

    manga_eye = Image.open("cartoon_feature/eye/eye.png")
    manga_nose = Image.open("cartoon_feature/nose/nose.png")
    manga_mouth = Image.open("cartoon_feature/mouth/mouth.png")
    manga_face = Image.open("cartoon_feature/face/face.png")
    manga_eyebrow = Image.open('output/option/' + gender + '_eyebrow.png')
    blank = Image.open('blank.jpg')


    manga_eye.thumbnail((eye.shape[1], eye.shape[1]), Image.ANTIALIAS)
    another_eye = manga_eye.transpose(Image.FLIP_LEFT_RIGHT)
    manga_eyebrow.thumbnail((eye.shape[1], eye.shape[1]), Image.ANTIALIAS)
    another_eyebrow = manga_eyebrow.transpose(Image.FLIP_LEFT_RIGHT)
    manga_nose.thumbnail((nose.shape[1], nose.shape[1]), Image.ANTIALIAS)
    manga_mouth.thumbnail((mouth.shape[1], mouth.shape[1]), Image.ANTIALIAS)
    manga_face = manga_face.resize((width, height), Image.ANTIALIAS)
    blank = blank.resize((width*2, height*2), Image.ANTIALIAS)

    manga_face_copy = manga_face.copy()

    eye_position = (x_eye, int(y_eye - manga_eye.size[1]/2))
    another_eye_position = (width - x_eye - manga_eye.size[0],int(y_eye - manga_eye.size[1]/2))
    eyebrow_position = (x_eye, int(y_eye - manga_eye.size[1]*1.5))
    another_eyebrow_position = (width - x_eye - manga_eye.size[0], int(y_eye - manga_eye.size[1]*1.5))
    nose_position = (int(width/2) - int(nose.shape[1]/2), int(y_nose - manga_nose.size[1]/2))
    mouth_position = (int(width/2) - int(mouth.shape[1]/2), int(y_mouth - manga_mouth.size[1]/2))

    manga_face_copy.paste(manga_eye, eye_position, manga_eye)
    manga_face_copy.paste(another_eye, another_eye_position, another_eye)
    manga_face_copy.paste(manga_eyebrow, eyebrow_position, manga_eyebrow)
    manga_face_copy.paste(another_eyebrow, another_eyebrow_position, another_eyebrow)
    manga_face_copy.paste(manga_nose, nose_position, manga_nose)
    manga_face_copy.paste(manga_mouth, mouth_position, manga_mouth)
    blank.paste(manga_face_copy,(int(width*0.5),int(height*0.75)),manga_face_copy)

    blank.save('pasted_image.jpg')

def hair():
    face = Image.open("pasted_image.jpg")
    manga_hair = Image.open("output/option/1.png")
    #glasses = Image.open("output/option/6.png")
    manga_hair.thumbnail((int(face.size[0]*0.55),int(face.size[0]*0.55)), Image.ANTIALIAS)
    #glasses.thumbnail((int(face.size[0] * 0.55), int(face.size[0] * 0.55)), Image.ANTIALIAS)
    face.paste(manga_hair, ((int(face.size[0]/2 - manga_hair.size[0]/2)), int(face.size[1]/15)), manga_hair)
    #face.paste(glasses, ((int(face.size[0] / 2 - glasses.size[0] / 2)), int(face.size[1]*0.4)), glasses)
    face.save('completed_image.jpg')

if __name__ == '__main__':
    age = "teen"
    gender = "boy"
    #webcam()
    #facial_landmarks()
    #overlay(gender)
    hair()