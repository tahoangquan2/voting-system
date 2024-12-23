import csv
from collections import Counter
import os
from difflib import SequenceMatcher

def MED(seq1, seq2):
    cache = [[float("inf")] * (len(seq2) + 1) for _ in range(len(seq1) + 1)]
    ops = [[None] * (len(seq2) + 1) for _ in range(len(seq1) + 1)]

    for j in range(len(seq2) + 1):
        cache[len(seq1)][j] = len(seq2) - j
        ops[len(seq1)][j] = 'insert'
    for i in range(len(seq1) + 1):
        cache[i][len(seq2)] = len(seq1) - i
        ops[i][len(seq2)] = 'delete'

    for i in range(len(seq1) - 1, -1, -1):
        for j in range(len(seq2) - 1, -1, -1):
            if seq1[i] == seq2[j]:
                cache[i][j] = cache[i + 1][j + 1]
                ops[i][j] = 'match'
            else:
                insert_cost = 1 + cache[i][j + 1]
                delete_cost = 1 + cache[i + 1][j]
                substitute_cost = 1 + cache[i + 1][j + 1]

                min_cost = min(insert_cost, delete_cost, substitute_cost)
                cache[i][j] = min_cost

                if min_cost == substitute_cost:
                    ops[i][j] = 'substitute'
                elif min_cost == delete_cost:
                    ops[i][j] = 'delete'
                else:
                    ops[i][j] = 'insert'

    aligned_seq1, aligned_seq2 = [], []
    i, j = 0, 0
    while i < len(seq1) or j < len(seq2):
        if i < len(seq1) and j < len(seq2) and ops[i][j] == 'match':
            aligned_seq1.append(seq1[i])
            aligned_seq2.append(seq2[j])
            i += 1
            j += 1
        elif i < len(seq1) and ops[i][j] == 'delete':
            aligned_seq1.append(seq1[i])
            aligned_seq2.append('`')
            i += 1
        elif j < len(seq2) and ops[i][j] == 'insert':
            aligned_seq1.append('`')
            aligned_seq2.append(seq2[j])
            j += 1
        elif i < len(seq1) and j < len(seq2) and ops[i][j] == 'substitute':
            aligned_seq1.append(seq1[i])
            aligned_seq2.append(seq2[j])
            i += 1
            j += 1

    return ''.join(aligned_seq1), ''.join(aligned_seq2)

def align_multiple_sequences(sequences):
    if not sequences:
        return []

    max_attempts = 3
    attempt = 0

    while attempt < max_attempts:
        try:
            first_seq = sequences[0]
            aligned_sequences = []

            for seq in sequences[1:]:
                new_seq, aligned_first = MED(seq, first_seq)
                first_seq = aligned_first
                aligned_sequences.append(new_seq)

            all_same_length = all(len(seq) == len(first_seq) for seq in aligned_sequences)
            if all_same_length:
                aligned_sequences.insert(0, first_seq)
                return aligned_sequences

            attempt += 1
            if attempt < max_attempts:
                sequences.append(sequences.pop(0))
        except:
            attempt += 1
            if attempt < max_attempts:
                sequences.append(sequences.pop(0))

    return sequences

def vote_characters(aligned_outputs):
    voted_result = []
    for i in range(len(aligned_outputs[0])):
        chars_at_pos = [output[i] for output in aligned_outputs]
        char_counts = Counter(chars_at_pos)
        most_common = char_counts.most_common()

        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            char1, count1 = most_common[0]
            char2, count2 = most_common[1]

            if char1 == '`' and char2 == ' ':
                voted_result.append('`')
            elif char2 == '`' and char1 == ' ':
                voted_result.append('`')
            elif char1 == '`':
                voted_result.append(char2)
            elif char2 == '`':
                voted_result.append(char1)
            else:
                voted_result.append(char1)
        else:
            voted_result.append(most_common[0][0])

    return ''.join(voted_result)

def refine_output(output):
    return output.replace('`', '')

def save_batch(output_file, batch_data):
    mode = 'a' if os.path.exists(output_file) else 'w'
    with open(output_file, mode, encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(batch_data)

def process_ocr_results(input_file='input.csv', output_file='output.csv'):
    batch_size = 100
    current_batch = []

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, 1):
            if len(row) < 2:
                continue

            file_name = row[0]
            ocr_results = row[1:]

            if len(ocr_results) < 2:
                continue

            try:
                # aligned_sequences = align_multiple_sequences(ocr_results)
                aligned_sequences = ocr_results
                if not aligned_sequences:
                    continue

                voted_output = vote_characters(aligned_sequences)
                final_output = refine_output(voted_output)

                new_row = [file_name] + ocr_results + [final_output]
                current_batch.append(new_row)

                if len(current_batch) >= batch_size:
                    save_batch(output_file, current_batch)
                    current_batch = []
                    print(f"Processed and saved {i} lines")
            except Exception as e:
                print(f"Error processing row {i}: {str(e)}")
                continue

    if current_batch:
        save_batch(output_file, current_batch)
        print(f"Processed and saved final batch")

if __name__ == "__main__":
    process_ocr_results()
