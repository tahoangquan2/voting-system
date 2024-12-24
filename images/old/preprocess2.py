import os
import cv2
import pytesseract
from pathlib import Path

def process_image(input_path, output_dir):
    img = cv2.imread(str(input_path))
    height, width = img.shape[:2]

    # ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang='vie')

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    word_count = 1

    for i in range(len(ocr_data['text'])):
        if not ocr_data['text'][i].strip():
            continue

        x = ocr_data['left'][i]
        y = ocr_data['top'][i]
        w = ocr_data['width'][i]
        h = ocr_data['height'][i]

        x1 = max(0, x - 2)
        y1 = max(0, y - 2)
        x2 = min(width, x + w + 2)
        y2 = min(height, y + h + 2)

        word_img = img[y1:y2, x1:x2]

        base_name = input_path.stem
        output_filename = f"{base_name}_{word_count:02d}.jpg"
        output_path = output_dir / output_filename

        cv2.imwrite(str(output_path), word_img)
        word_count += 1

def main():
    input_dir = Path('output')
    output_dir = Path('output2')

    for image_path in sorted(input_dir.glob('*.jpg')):
        process_image(image_path, output_dir)

if __name__ == "__main__":
    main()
