import os
import csv
import time
from PIL import Image
import pytesseract

def process_images(input_folder, output_csv):
    pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'

    results = []
    total_files = len([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))])
    start_time = time.time()
    processed = 0

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            try:
                print(f"Processing {filename} ({processed + 1}/{total_files})")
                image_start = time.time()

                image = Image.open(image_path)
                text = pytesseract.image_to_string(image, lang='vie')
                results.append([filename, text.strip()])

                image_time = time.time() - image_start
                print(f"Completed in {image_time:.2f} seconds")

                processed += 1
                if processed % 100 == 0:
                    save_results(results, output_csv, processed == total_files)
                    results = []
                    print(f"Saved batch of {processed} images")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                results.append([filename, f"Error: {str(e)}"])

    if results:
        save_results(results, output_csv, False)

    total_time = time.time() - start_time
    print(f"\nCompleted processing {processed} images in {total_time:.2f} seconds")
    print(f"Average time per image: {total_time/processed:.2f} seconds")

def save_results(results, output_csv, first_batch):
    mode = 'w' if first_batch else 'a'
    with open(output_csv, mode, newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if first_batch:
            writer.writerow(['filename', 'text'])
        writer.writerows(results)

if __name__ == '__main__':
    input_folder = 'input'
    output_csv = 'output.csv'
    process_images(input_folder, output_csv)
