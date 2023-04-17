# Uncomment code below and install these packages

# pip install PyPDF2
# pip install PyMuPDF
# pip install pdf2image
# apt-get install poppler-utils
# sudo apt install tesseract-ocr
# pip install pytesseract
# sudo apt-get install tesseract-ocr-dan
# pip install fastapi uvicorn

import io
import os
import re
import cv2
import glob
import joblib
import string
import uvicorn
import tempfile
import subprocess
import pytesseract
import numpy as np
from typing import List, Union
from collections import OrderedDict
from keras.models import load_model
from keras_contrib.layers import CRF
from keras.utils import pad_sequences
from pdf2image import convert_from_path
from keras_contrib.losses import crf_loss
from fastapi.responses import JSONResponse
from pdfminer.high_level import extract_text
from fastapi import FastAPI, File, UploadFile
from keras_contrib.metrics import crf_viterbi_accuracy

app = FastAPI()

# Code of Vision_pipeline
def pdf_to_image(pdf_name, image_folder_path):

    image_folder_name = os.path.splitext(os.path.basename(pdf_name))[0]
    image_folder_path = os.path.join(image_folder_path, image_folder_name)

    # Create the image folder if it doesn't exist
    if not os.path.exists(image_folder_path):
        os.makedirs(image_folder_path)

    # Convert the PDF file to JPEG images and save them to the image folder
    images = convert_from_path(pdf_name, dpi=300, fmt='jpg')

    for i, image in enumerate(images):
        # Construct the filename using the PDF filename and page number
        filename = f'{os.path.splitext(os.path.basename(pdf_name))[0]}_{i+1}.jpg'
        image_path = os.path.join(image_folder_path, filename)
        # Save the image to the specified path
        image.save(image_path, 'JPEG')

    return image_folder_name


def run_detection(weights_path, conf_threshold, source_path, yolo_folder):
    cmd = f'python {yolo_folder}/detect.py --weights {weights_path} --conf {conf_threshold} --source {source_path} --save-txt'
    subprocess.run(cmd, shell=True)


def get_annotations(yolo_folder):
    # set the parent folder path
    parent_folder = os.path.join(yolo_folder, 'runs', 'detect')

    # get the subfolders in the parent folder
    subfolders = os.listdir(parent_folder)

    # exclude the default first subfolder (if present)
    subfolders = [subfolder for subfolder in subfolders if subfolder != 'exp']

    # extract the numerical values from the remaining subfolder names
    subfolder_numbers = [int(subfolder.split('exp')[-1])
                         for subfolder in subfolders if 'exp' in subfolder and subfolder.split('exp')[-1].isdigit()]

    if subfolder_numbers:
        # get the latest subfolder based on its numerical value
        latest_subfolder_number = max(subfolder_numbers)
        latest_subfolder = f'exp{latest_subfolder_number}'
    else:
        # use the default subfolder if no 'exp' subfolders found
        latest_subfolder = 'exp'

    # construct the annotation folder path
    annotation_folder = os.path.join(parent_folder, latest_subfolder, 'labels')
    return annotation_folder


def extract_data(pdf_images, annotation_folder):

    nummer = []
    belob = []
    total = []
    cvr = []
    sum = 0

    img_paths = sorted(glob.glob(pdf_images + '/*.jpg'))

    # loop over all images in the folder
    for img_path in img_paths:

        # load the image
        img = cv2.imread(img_path)
        dh, dw, _ = img.shape
        # _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # extract the file name from img_path
        file_name = os.path.basename(img_path)
        file_name = os.path.splitext(file_name)[0]

        # construct the annotation_path
        annotation_path = os.path.join(annotation_folder, file_name + '.txt')

        fl = open(annotation_path, "r")
        data = fl.readlines()
        fl.close()

        for dt in data:
            lab, x, y, w, h = map(float, dt.split(" "))
            label = int(lab)
            l = int((x - w / 2) * dw)
            r = int((x + w / 2) * dw)
            t = int((y - h / 2) * dh)
            b = int((y + h / 2) * dh)

            if l < 0:
                l = 0
            if r > dw - 1:
                r = dw - 1
            if t < 0:
                t = 0
            if b > dh - 1:
                b = dh - 1

            # extract the region of interest
            roi = img[t:b, l:r]

            # apply OCR to the ROI
            text = pytesseract.image_to_string(roi, lang="dan", config="--oem 3 --psm 4")

            if label == 0:
                for num in text.split('\n'):
                    # extract numeric values using regular expressions
                    nums = re.findall(r'\d+', num)
                    # append only the first numeric value found
                    if nums:
                        nummer.append(nums[0])
            elif label == 1:
                for amount in text.split('\n'):
                    match = re.search(r'\d+([,.]\d+)*', amount)
                    if match:
                        # replace dot with empty string, then comma with period
                        value = match.group().replace('.', '').replace(',', '.')
                        value_float = float(value)
                        if round(value_float, 2) != round(sum, 2):
                            belob.append(match.group())
                            sum += value_float

            elif label == 2:
                for tot in text.split('\n'):
                    if tot.strip():
                        total.append(
                            ''.join(filter(lambda x: x in string.printable, tot)).replace('\x0c', ''))
            elif label == 3:
                for cv in text.split('\n'):
                    if cv.strip():
                        cvr.append(
                            ''.join(filter(lambda x: x in string.printable, cv)).replace('\x0c', ''))

    for item in cvr:
        cvr_number = ''.join([char for char in item if char.isnumeric()])
    for item in total:
        # extract the first four elements
        total_split = ''.join(item.split(' ')[5:])

    # Return output as a dictionary
    return {"Nummer": nummer, "Beløb": belob}

# Code of Nlp_pipeline

max_len = 1000
model_predict = load_model(r'E:\VSCODE\visionapiflask\data\model.h5',custom_objects={"CRF": CRF, 'crf_loss': crf_loss, 'crf_viterbi_accuracy': crf_viterbi_accuracy})

# Load the word2idx and tag2idx dictionaries
with open(r'E:\VSCODE\visionapiflask\data\words.pkl','rb') as f:
    words = joblib.load(f)

with open(r'E:\VSCODE\visionapiflask\data\tags.pkl','rb') as f:
    tags = joblib.load(f)

word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {i: w for w, i in tag2idx.items()}
def test_sentence_sample(test_sentence):
    results = []
    x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in tokenize(test_sentence)]], padding="post",
                                value=0, maxlen=max_len)
    p = model_predict.predict(np.array([x_test_sent[0]]))
    p = np.argmax(p, axis=-1)
    for w, pred in zip(tokenize(test_sentence), p[0]):
        results.append([w, tags[pred]])
    return results

def tokenize(s):
    return s.split()
def process_text_file(input_text):
    # Initialize variables
    product_number = []
    product_vat = []
    total_amount = []
    cvrs = []

    # Loop through the lines in the input text
    for line in input_text:
        # Strip any leading/trailing whitespace
        line = [elem.strip() for elem in line]

        # If the line contains one of the desired keys
        if "B-TOTAL-AMOUNT" in line[1]:
            total_amount = line[0]

        elif "B-PRODUCT-VAT" in line[1]:
            product_vat = [line[0]]

        elif "I-PRODUCT-VAT" in line[1]:
            product_vat.append(line[0])

        elif "B-CVR" in line[1] or "I-CVR" in line[1]:
            cvrs.append(line[0])

        elif "B-PRODUCT-NUMBER" in line[1] or "I-PRODUCT-NUMBER" in line[1]:
            product_number.append(line[0])

    # Return output as a dictionary
    return {"I alt DKK inkl. moms": total_amount,
            "CVR": ''.join(list(OrderedDict.fromkeys(cvrs)))}

@app.post("/predict")
async def main(files: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file_contents = await files.read()  # Read file contents once
        tmp.write(file_contents)  # Write to temporary file
        tmp.flush()
    pdf_name = tmp.name

    image_folder_path = r"E:\VSCODE\visionapiflask\images"
    pdf_images = f'{image_folder_path}/{pdf_to_image(pdf_name, image_folder_path)}'
    weights_path = r'E:\VSCODE\visionapiflask\best.pt'
    yolo_folder = r'E:\VSCODE\visionapiflask\yolov5'
    conf_threshold = 0.9

    run_detection(weights_path, conf_threshold,pdf_images, yolo_folder)
    annotation_folder = get_annotations(yolo_folder)
    vision_result = extract_data(pdf_images, annotation_folder)

    # file_contents = files.read()
    # Extract text from the file
    text = extract_text(io.BytesIO(file_contents))

    # Predict tags for the text
    results = test_sentence_sample(text)

    # Process the text and write the output to a CSV file
    output = process_text_file(results)

    merged_data = {
            "Nummer": vision_result["Nummer"],
            "Beløb": vision_result["Beløb"],
            "I alt DKK inkl. moms": output["I alt DKK inkl. moms"],
            "CVR": output["CVR"]
    }
    print("Hello: ",merged_data)

    # Create a JSONResponse object from the merged data
    return JSONResponse(merged_data)

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)