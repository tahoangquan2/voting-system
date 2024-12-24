import cv2
import numpy as np
import pandas as pd
import os
import ast
from itertools import groupby

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    padding = 4
    padded_width = maxWidth + (2 * padding)
    padded_height = maxHeight + (2 * padding)

    dst = np.array([
        [padding, padding],
        [padded_width - padding - 1, padding],
        [padded_width - padding - 1, padded_height - padding - 1],
        [padding, padded_height - padding - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (padded_width, padded_height))
    return warped

df = pd.read_csv('input.csv', header=0, names=['page', 'bounding_box', 'confidence', 'source'])
os.makedirs('output', exist_ok=True)

for page_num, group in groupby(df.iterrows(), key=lambda x: x[1]['page']):
    image_path = os.path.join('images', f'page_{int(page_num)}.jpeg')
    print(f"Processing: {image_path}")

    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image: {image_path}")
        continue

    for crop_idx, (_, row) in enumerate(group, start=1):
        try:
            bbox = np.array(ast.literal_eval(row['bounding_box']), dtype="float32")
            warped = four_point_transform(image, bbox)
            output_path = os.path.join('output', f'page_{int(page_num):03d}_{crop_idx:02d}.jpg')
            cv2.imwrite(output_path, warped)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error processing crop {crop_idx} from page {page_num}: {str(e)}")
            continue
