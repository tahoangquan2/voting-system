import pandas as pd
import ast
import numpy as np

def calculate_polygon_area(points):
    """Calculate area of polygon using shoelace formula."""
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    return 0.5 * abs(sum(i * j for i, j in zip(x, y[1:] + [y[0]])) - 
                    sum(i * j for i, j in zip(x[1:] + [x[0]], y)))

def get_intersection_points(box1, box2):
    """Get intersection points between two polygons."""
    from shapely.geometry import Polygon, LineString
    
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    
    if not poly1.intersects(poly2):
        return []
        
    # Get the intersection
    intersection = poly1.intersection(poly2)
    
    if intersection.geom_type == 'Polygon':
        return list(intersection.exterior.coords)[:-1]  # Remove duplicate last point
    return []

def calculate_overlap_ratio(box1, box2):
    """
    Calculate the overlap ratio between two polygons using exact geometry.
    """
    try:
        # Get intersection points
        intersection_points = get_intersection_points(box1, box2)
        
        if not intersection_points:
            return 0.0
            
        # Calculate areas
        box1_area = calculate_polygon_area(box1)
        intersection_area = calculate_polygon_area(intersection_points)
        
        # Calculate ratio
        ratio = intersection_area / box1_area if box1_area > 0 else 0.0
        
        # Print debug information
        # print(f"Box1 area: {box1_area}")
        # print(f"Intersection area: {intersection_area}")
        # print(f"Ratio: {ratio}")
        
        return ratio
        
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
                    break
            
            if not matched and box1['confidence'] > 0.85:
                output_data.append({
                    'page': page,
                    'bounding_box': box1['bounding_box'],
                    'confidence': box1['confidence'],
                    'source': 'ocr1'
                })
        
        for idx2, box2 in page_boxes2.iterrows():
            if idx2 not in matched_boxes2 and box2['confidence'] > 0.85:
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