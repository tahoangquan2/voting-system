import pandas as pd
import ast

def calculate_iou_area(box1, box2):
    x1_min = min(point[0] for point in box1)
    y1_min = min(point[1] for point in box1)
    x1_max = max(point[0] for point in box1)
    y1_max = max(point[1] for point in box1)
    
    x2_min = min(point[0] for point in box2)
    y2_min = min(point[1] for point in box2)
    x2_max = max(point[0] for point in box2)
    y2_max = max(point[1] for point in box2)
    
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    
    return intersection / area1 if area1 > 0 else 0

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
                overlap_ratio = calculate_iou_area(box1_coords, box2_coords)
                
                if overlap_ratio > 0.7:
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
            
            if not matched and box1['confidence'] > 0.5:
                output_data.append({
                    'page': page,
                    'bounding_box': box1['bounding_box'],
                    'confidence': box1['confidence'],
                    'source': 'ocr1'
                })
        
        for idx2, box2 in page_boxes2.iterrows():
            if idx2 not in matched_boxes2 and box2['confidence'] > 0.5:
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
        'paddleENGbbconfi.csv',
        'Surya.csv',
        'matched_boxes.csv'
    )