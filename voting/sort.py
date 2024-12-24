import csv
from operator import itemgetter

def sort_bounding_boxes(input_csv, output_csv):
    pages = {}

    # Read and parse the CSV
    with open(input_csv, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            page = int(row['page'])
            # Convert string bounding box to list of coordinates
            bbox = eval(row['bounding_box'])
            confidence = float(row['confidence'])
            source = row['source']

            # Get the y-coordinate of the top-left corner for sorting
            y_coord = bbox[0][1]  # [0][1] gives us the y-coordinate of the first point

            # Store all information
            box_info = {
                'page': page,
                'bbox': bbox,
                'confidence': confidence,
                'source': source,
                'y_coord': y_coord
            }

            # Add to pages dictionary
            if page not in pages:
                pages[page] = []
            pages[page].append(box_info)

    # Sort boxes in each page by y-coordinate
    for page in pages:
        pages[page].sort(key=itemgetter('y_coord'))

    # Write sorted results to output CSV
    with open(output_csv, 'w', newline='') as file:
        fieldnames = ['page', 'bounding_box', 'confidence', 'source']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header
        writer.writeheader()

        for page in sorted(pages.keys()):
            for box in pages[page]:
                writer.writerow({
                    'page': box['page'],
                    'bounding_box': box['bbox'],
                    'confidence': box['confidence'],
                    'source': box['source']
                })

if __name__ == "__main__":
    input_file = "matched_boxes.csv"
    output_file = "sorted_output.csv"
    sort_bounding_boxes(input_file, output_file)
