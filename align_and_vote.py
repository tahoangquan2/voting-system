import csv
from collections import Counter
import pandas as pd
import re


def normalize_text(text):
    # Remove newline characters
    text = re.sub(r'\s*[\n\r]+\s*', ' ', text)

    # Convert "~" and "=" to "-"
    text = text.replace("~", "-").replace("=", "-")

    # Convert "«»" and to '""'
    text = text.replace("« ", '"').replace(" »", '"')

    # Ensure a space follows "-" at the start of the text
    text = re.sub(r'\.\.\.', '…', text)

    # Ensure punctuation is followed by a space
    text = re.sub(r'([….,;:?!])\s', r'\1 ', text)

    # Remove spaces before punctuation
    text = re.sub(r'\s+([….,!?;:])', r'\1', text)

    # Ensure only one space between words
    text = re.sub(r'\s+', ' ', text)

    # Ensure a space follows "-" at the start of the text
    text = re.sub(r'^-\s*', '- ', text)

    # Strip leading and trailing spaces
    text = text.strip()

    # Return normalized text
    return text


def MED(seq1, seq2):
    # Initialize cache for MED calculation and operations tracking
    cache = [[float("inf")] * (len(seq2) + 1) for _ in range(len(seq1) + 1)]
    ops = [[None] * (len(seq2) + 1) for _ in range(len(seq1) + 1)]

    # Fill base cases
    for j in range(len(seq2) + 1):
        cache[len(seq1)][j] = len(seq2) - j
        ops[len(seq1)][j] = 'insert'  # Need to insert remaining qn chars
    for i in range(len(seq1) + 1):
        cache[i][len(seq2)] = len(seq1) - i
        ops[i][len(seq2)] = 'delete'  # Need to delete remaining seq1 chars

    # Fill the cache and ops table
    for i in range(len(seq1) - 1, -1, -1):
        for j in range(len(seq2) - 1, -1, -1):

            if seq1[i] == seq2[j]:
                cache[i][j] = cache[i + 1][j + 1]
                ops[i][j] = 'match'  # Characters match, move diagonally
            else:
                # Consider all operations: insert, delete, substitute
                insert_cost = 1 + cache[i][j + 1]
                delete_cost = 1 + cache[i + 1][j]
                substitute_cost = 1 + cache[i + 1][j + 1]

                # Choose the operation with the minimum cost
                if insert_cost <= delete_cost and insert_cost <= substitute_cost:
                    cache[i][j] = insert_cost
                    ops[i][j] = 'insert'
                elif delete_cost <= insert_cost and delete_cost <= substitute_cost:
                    cache[i][j] = delete_cost
                    ops[i][j] = 'delete'
                else:
                    cache[i][j] = substitute_cost
                    ops[i][j] = 'substitute'

    # Backtrack
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

    aligned_seq1 = ''.join(aligned_seq1)
    aligned_seq2 = ''.join(aligned_seq2)
    return aligned_seq1, aligned_seq2


def align_multiple_sequences(sequences):
    try_case = 0
    while try_case < 4:
        aligned_sequences = sequences[:]
        limit = 0
        while limit < 3:
            # Sort sequences by length (descending) to align shorter ones with the longest
            aligned_sequences.sort(key=len, reverse=True)
            if limit == 0:
                longest_sequence = aligned_sequences.pop(0)
            else:
                longest_sequence = aligned_sequences.pop(try_case)

            # Align each remaining sequence to the longest sequence
            updated_sequences = []
            for seq in aligned_sequences:
                aligned_seq, longest_sequence = MED(seq, longest_sequence)
                updated_sequences.append(aligned_seq)

            # Update the list of aligned sequences
            aligned_sequences = updated_sequences + [longest_sequence]

            # Check if all sequences are aligned to the same length
            if all(len(seq) == len(longest_sequence) for seq in aligned_sequences):
                return aligned_sequences

            # print(f"try_case {try_case} - limit {limit}:")
            # for seq in aligned_sequences:
            #     print(seq)

            limit += 1

        try_case += 1
        # print(f"Try case {try_case} done")


def read_ocr_inputs(csv_file):
    ocr_outputs = []
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            # Skip the filename column and extract the next 4 columns
            ocr_outputs.append(row[1:5])  # Columns 2 to 5 (0-based index)

    return ocr_outputs


def vote_characters(aligned_outputs):
    # Perform voting for each character position
    voted_result = []
    for i in range(len(aligned_outputs[0])):
        # Gather characters at the current position
        chars_at_pos = [output[i] for output in aligned_outputs]

        # Count character frequencies
        char_counts = Counter(chars_at_pos)

        # Extract the most frequent characters
        most_common = char_counts.most_common()

        # Apply tiebreaker rules:
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:  # A tie exists
            char1, count1 = most_common[0]
            char2, count2 = most_common[1]

            if char1 == '`' and char2 == ' ':
                voted_result.append('`')  # Backquote beats space
            elif char2 == '`' and char1 == ' ':
                voted_result.append('`')  # Backquote beats space
            elif char1 == '`':
                voted_result.append(char2)  # Other character beats backquote
            elif char2 == '`':
                voted_result.append(char1)  # Other character beats backquote
            else:
                voted_result.append(char1)  # Default to the first one if no special cases
        else:
            # No tie or clear winner, choose the most common character
            voted_result.append(most_common[0][0])

    # Return the final corrected string
    return ''.join(voted_result)


def refine_output(output):
    output = output.replace('`', '')
    return output


def add_voted_results(csv_file, voted_results):
    file_path = csv_file
    df = pd.read_csv(file_path, header=None, encoding='utf-8')

    # Step 3: Add the new column
    df[len(df.columns)] = voted_results  # Adds the list as a new column

    # Step 4: Save the updated DataFrame back to a CSV file
    df.to_csv('output.csv', index=False, header=False, encoding='utf-8')
    df.to_excel('output.xlsx', index=False, header=False)

    print(f"New column added")


if __name__ == "__main__":
    # # Debug for specific row
    # ocr_outputs = read_ocr_inputs('input.csv')[260]
    # norm_ocr_outputs = []
    # for each in ocr_outputs:
    #     each = normalize_text(each)
    #     norm_ocr_outputs.append(each)
    # # print(norm_ocr_outputs)

    # aligned_result = align_multiple_sequences(norm_ocr_outputs)
    # for seq in aligned_result:
    #     print(f"'{seq}',")

    final_ouputs = []
    ocr_outputs = read_ocr_inputs('input.csv')
    for index, line in enumerate(ocr_outputs):
        print(f"[!] Processing row {index+1}")

        norm_ocr_outputs = []
        for each in line:
            norm_ocr_outputs.append(normalize_text(each))

        aligned_result = align_multiple_sequences(norm_ocr_outputs)
        for seq in aligned_result:
            print(f'"{seq}",')

        voted_output = vote_characters(aligned_result)
        final_ouputs.append(refine_output(voted_output))

    add_voted_results('input.csv', final_ouputs)
