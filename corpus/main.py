import pandas as pd
import Levenshtein
import re

def read_csv_safely(file_path):
    return pd.read_csv(file_path, encoding='utf-8', dtype=str)

def preprocess_page_format(page_str, format_type='a'):
    if not isinstance(page_str, str):
        return page_str

    if format_type == 'a':
        match = re.search(r'page(\d+)', page_str)
        if match:
            page_num = str(int(match.group(1)))
            return page_num
    else:
        match = re.search(r'page_(\d+)_\d+\.jpg', page_str)
        if match:
            page_num = str(int(match.group(1)))
            return page_num

    return page_str

def calculate_similarity(str1, str2):
    if not isinstance(str1, str) or not isinstance(str2, str):
        return 0
    if not str1 or not str2:  # Handle empty strings
        return 0
    distance = Levenshtein.distance(str1, str2)
    max_len = max(len(str1), len(str2))
    return (max_len - distance) / max_len * 100

def analyze_ocr_tool(df_a, df_b, ocr_column, similarity_threshold=70):
    total_matches = 0
    total_similarity = 0
    total_b_lines = len(df_b)
    matched_b_indices = set()

    unique_pages = sorted(set(df_a['page'].unique()) | set(df_b['page'].unique()))

    matches = []

    for page in unique_pages:
        page_a = df_a[df_a['page'] == page]
        page_b = df_b[df_b['page'] == page]

        # For each line in A (correct text), find best match in B
        for idx_a, row_a in page_a.iterrows():
            best_match = None
            best_similarity = -1
            best_idx_b = None

            # Compare with each unmatched line in B
            for idx_b, row_b in page_b.iterrows():
                if idx_b in matched_b_indices:
                    continue

                similarity = calculate_similarity(row_a['text'], row_b[ocr_column])

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = row_b
                    best_idx_b = idx_b

            # If we found a good match
            if best_similarity >= similarity_threshold and best_idx_b is not None:
                matched_b_indices.add(best_idx_b)
                total_matches += 1
                total_similarity += best_similarity
                # print(f"Matched: {row_a['text']} -> {best_match[ocr_column]} ({best_similarity}%)")

                matches.append({
                    'page': page,
                    'box': row_a['box'],
                    'ground_truth': row_a['text'],
                    'ocr_text': best_match[ocr_column],
                    'similarity': round(best_similarity, 2)
                })

    # Calculate statistics
    skipped_lines = total_b_lines - total_matches
    average_similarity = (total_similarity / total_matches) if total_matches > 0 else 0

    return {
        'accuracy_percentage': round(average_similarity, 2),
        'total_skipped_lines': skipped_lines,
        'total_matched_lines': total_matches,
        'total_b_lines': total_b_lines,
        'matching_percentage': round((total_matches / total_b_lines * 100), 2) if total_b_lines > 0 else 0,
        'matches': matches
    }

def analyze_all_ocr_tools(csv_a_path, csv_b_path, output_path, similarity_threshold=30):
    df_a = read_csv_safely(csv_a_path)
    df_b = read_csv_safely(csv_b_path)

    df_a['page'] = df_a['page'].apply(lambda x: preprocess_page_format(x, format_type='a'))
    df_b['page'] = df_b['page'].apply(lambda x: preprocess_page_format(x, format_type='b'))

    ocr_columns = df_b.columns[1:]

    with open(output_path, 'w', encoding='utf-8') as f:
        for ocr_column in ocr_columns:
            print(f"Analyzing OCR tool: {ocr_column}")
            results = analyze_ocr_tool(df_a, df_b, ocr_column, similarity_threshold)

            f.write(f"{ocr_column}:\n")
            f.write(f"Accuracy: {results['accuracy_percentage']}%\n")
            f.write(f"Match rate: {results['matching_percentage']}%\n")
            f.write(f"Matched/Total: {results['total_matched_lines']}/{results['total_b_lines']}\n")
            f.write(f"Skipped: {results['total_skipped_lines']}\n\n")

if __name__ == "__main__":
    analyze_all_ocr_tools('a.csv', 'b.csv', 'ocr_comparison_results.txt')
