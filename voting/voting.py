import pandas as pd
import ast
from shapely.geometry import Polygon

def calculate_overlap_ratio(box1, box2):
    try:
        polygon1 = Polygon(box1)
        polygon2 = Polygon(box2)

        if not polygon1.is_valid or not polygon2.is_valid:
            return 0.0

        intersection = polygon1.intersection(polygon2)

        area1 = polygon1.area
        area2 = polygon2.area
        intersection_area = intersection.area

        ratio1 = intersection_area / area1 if area1 > 0 else 0.0
        ratio2 = intersection_area / area2 if area2 > 0 else 0.0

        max_ratio = max(ratio1, ratio2)

        # print(f"Box1 area: {area1}")
        # print(f"Box2 area: {area2}")
        # print(f"Intersection area: {intersection_area}")
        # print(f"Ratio1 (intersection/box1): {ratio1}")
        # print(f"Ratio2 (intersection/box2): {ratio2}")
        # print(f"Max ratio: {max_ratio}")

        return max_ratio

    except Exception as e:
        print(f"Error calculating overlap: {e}")
        return 0.0

def parse_bbox(bbox_str):
    return ast.literal_eval(bbox_str)

def compare_ocr_results(file1_path, file2_path, output_path):
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    output_data = []

    for page in df1['page'].unique():
        page_boxes1 = df1[df1['page'] == page]
        page_boxes2 = df2[df2['page'] == page]
        matched_boxes2 = set()

        for _, box1 in page_boxes1.iterrows():
            box1_coords = parse_bbox(box1['bounding_box'])
            matched = False

            for idx2, box2 in page_boxes2.iterrows():
                if idx2 in matched_boxes2:
                    continue

                box2_coords = parse_bbox(box2['bounding_box'])
                overlap_ratio = calculate_overlap_ratio(box1_coords, box2_coords)

                if overlap_ratio >= 0.4:
                    avg_confidence = (box1['confidence'] + box2['confidence']) / 2
                    if avg_confidence > 0.5:
                        output_data.append({
                            'page': page,
                            'bounding_box': box1['bounding_box'],
                            'confidence': avg_confidence,
                            'source': 'matched'
                        })
                    matched_boxes2.add(idx2)
                    matched = True

            if not matched and box1['confidence'] > 0.8:
                output_data.append({
                    'page': page,
                    'bounding_box': box1['bounding_box'],
                    'confidence': box1['confidence'],
                    'source': 'ocr1'
                })

        for idx2, box2 in page_boxes2.iterrows():
            if idx2 not in matched_boxes2 and box2['confidence'] > 0.8:
                output_data.append({
                    'page': page,
                    'bounding_box': box2['bounding_box'],
                    'confidence': box2['confidence'],
                    'source': 'ocr2'
                })

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_path, index=False)
    print(f"Processed results saved to {output_path}")

if __name__ == "__main__":

    compare_ocr_results(
        'paddleENG_270.csv',
        'Surya_270.csv',
        'matched_boxes.csv'
    )
