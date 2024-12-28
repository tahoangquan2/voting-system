import pandas as pd
import Levenshtein
import re

def read_csv_safely(file_path):
    return pd.read_csv(file_path, encoding='utf-8', dtype=str)

def preprocess_page_format(page_str):
    if not isinstance(page_str, str):
        return page_str

    # Extract the number using regex
    match = re.search(r'page(\d+)', page_str)
    if match:
        # Convert to integer and back to string to remove leading zeros
        page_num = str(int(match.group(1)))
        return page_num
    return page_str

def calculate_similarity(str1, str2):
    if not str1 or not str2:  # Handle empty strings
        return 0
    distance = Levenshtein.distance(str1, str2)
    max_len = max(len(str1), len(str2))
    return (max_len - distance) / max_len * 100

def match_and_analyze_csvs(csv_a_path, csv_b_path, similarity_threshold=70):
    df_a = read_csv_safely(csv_a_path)
    df_b = read_csv_safely(csv_b_path)
    df_a['page'] = df_a['page'].apply(preprocess_page_format)

    total_matches = 0
    total_similarity = 0
    total_b_lines = len(df_b)
    matched_b_indices = set()

    unique_pages = sorted(set(df_a['page'].unique()) | set(df_b['page'].unique()))

    for page in unique_pages:
        page_a = df_a[df_a['page'] == page]
        page_b = df_b[df_b['page'] == page]

        # For each line in A (correct text), find best match in B (complete boxes)
        for idx_a, row_a in page_a.iterrows():
            best_match = None
            best_similarity = -1
            best_idx_b = None

            # Compare with each unmatched line in B
            for idx_b, row_b in page_b.iterrows():
                if idx_b in matched_b_indices:
                    continue

                similarity = calculate_similarity(row_a['text'], row_b['text'])

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = row_b
                    best_idx_b = idx_b

            # If we found a good match
            if best_similarity >= similarity_threshold and best_idx_b is not None:
                matched_b_indices.add(best_idx_b)
                total_matches += 1
                total_similarity += best_similarity

    # Calculate statistics
    skipped_lines = total_b_lines - total_matches
    average_similarity = (total_similarity / total_matches) if total_matches > 0 else 0

    return {
        'accuracy_percentage': round(average_similarity, 2),
        'total_skipped_lines': skipped_lines,
        'total_matched_lines': total_matches,
        'total_b_lines': total_b_lines,
        'matching_percentage': round((total_matches / total_b_lines * 100), 2) if total_b_lines > 0 else 0
    }

if __name__ == "__main__":
    results = match_and_analyze_csvs('a.csv', 'b.csv')
    print(f"Text accuracy of matched lines: {results['accuracy_percentage']}%")
    print(f"Total lines in B: {results['total_b_lines']}")
    print(f"Successfully matched lines: {results['total_matched_lines']} ({results['matching_percentage']}%)")
    print(f"Skipped/unmatched lines in B: {results['total_skipped_lines']}")
