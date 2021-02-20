# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 08:44:02 2021

@author: yerminal
@website: https://github.com/yerminal
"""
"""
For those who don't want to work on this,
Tutorial Link: https://medium.com/@ahmetxgenc/how-to-use-tesseract-on-windows-fe9d2a9ba5c6
Tesseract Download Link: https://digi.bib.uni-mannheim.de/tesseract/
"""
import pytesseract
from PIL import Image
import cv2

file_path= "test.jpg"
im = Image.open(file_path)
im.save("ocr.png", dpi=(300, 300))

image = cv2.imread("ocr.png")
image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
retval, threshold = cv2.threshold(image,127,255,cv2.THRESH_BINARY)

text = pytesseract.image_to_string(threshold)

with open("output.txt", "w",5 ,"utf-8") as text_file: 
    text_file.write(text)