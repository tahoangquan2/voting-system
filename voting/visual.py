import cv2
import pandas as pd
import numpy as np
import ast
import os
from typing import List, Tuple

class BoundingBoxVisualizer:
    def __init__(self, csv_path: str, images_folder: str):
        """
        Initialize the BoundingBoxVisualizer.

        Args:
            csv_path (str): Path to the CSV file containing bounding box data
            images_folder (str): Path to the folder containing images
        """
        self.df = pd.read_csv(csv_path)
        self.images_folder = images_folder

        # Color mapping for different sources (in BGR format for OpenCV)
        self.colors = {
            'matched': (0, 255, 0),  # Green
            'ocr1': (0, 0, 255),     # Red
            'ocr2': (255, 0, 0)      # Blue
        }

    def parse_bounding_box(self, bbox_str: str) -> List[Tuple[int, int]]:
        """
        Parse the bounding box string into a list of coordinates.

        Args:
            bbox_str (str): String representation of bounding box coordinates

        Returns:
            List[Tuple[int, int]]: List of (x, y) coordinates
        """
        try:
            # Convert string to Python list using ast.literal_eval
            coords = ast.literal_eval(bbox_str)
            # Convert to list of tuples
            return [(int(x), int(y)) for x, y in coords]
        except Exception as e:
            print(f"Error parsing bounding box: {e}")
            return []

    def draw_bounding_box(self, image, coords: List[Tuple[int, int]], color: Tuple[int, int, int], index: int):
        """
        Draw a bounding box on the image and label it with an index number.

        Args:
            image: OpenCV image
            coords (List[Tuple[int, int]]): List of corner coordinates
            color (Tuple[int, int, int]): BGR color tuple
            index (int): Index number to label the bounding box
        """
        # Convert coordinates to numpy array
        pts = np.array(coords, np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Draw the polygon
        cv2.polylines(image, [pts], True, color, 2)

        # Calculate the position for the index label (top-left corner of the bounding box)
        label_position = (pts[0][0][0], pts[0][0][1] - 10)

        # Draw the index label
        cv2.putText(image, str(index), label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def process_page(self, page_number: int, output_folder: str = "output"):
        """
        Process a single page and draw all bounding boxes.

        Args:
            page_number (int): Page number to process
            output_folder (str): Folder to save the output image
        """
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Load image
        image_path = os.path.join(self.images_folder, f"page_{page_number}.jpeg")
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            return

        # Get bounding boxes for this page
        page_boxes = self.df[self.df['page'] == page_number]

        # Draw each bounding box
        for _, row in page_boxes.iterrows():
            coords = self.parse_bounding_box(row['bounding_box'])
            if coords:
                color = self.colors.get(row['source'], (0, 255, 255))  # Default to yellow if source not found
                self.draw_bounding_box(image, coords, color, _)

        # Save the output image
        output_path = os.path.join(output_folder, f"page_{page_number}_annotated.jpeg")
        cv2.imwrite(output_path, image)
        print(f"Processed page {page_number}, saved to {output_path}")

    def process_all_pages(self, output_folder: str = "output"):
        """
        Process all pages in the dataset.

        Args:
            output_folder (str): Folder to save the output images
        """
        unique_pages = self.df['page'].unique()
        for page in unique_pages:
            self.process_page(page, output_folder)

def main():
    # Example usage
    csv_path = "sorted_output.csv"  # Path to your CSV file
    images_folder = "images"       # Path to folder containing images
    output_folder = "output"       # Path to save annotated images

    visualizer = BoundingBoxVisualizer(csv_path, images_folder)

    # Process all pages
    visualizer.process_all_pages(output_folder)

    # Or process a specific page
    # visualizer.process_page(1, output_folder)

if __name__ == "__main__":
    main()
