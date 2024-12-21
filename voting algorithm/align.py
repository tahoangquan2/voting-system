import numpy as np


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


def align_multiple_sequences(sequsences):
    first_seq = sequsences[0]
    aligned_sequences = sequsences[1:]
    count = 0
    while count != len(aligned_sequences):
        count = 0
        aligned_sequences_new = []
        for seq in aligned_sequences:
            new_seq, first_seq = MED(seq, first_seq)
            aligned_sequences_new.append(new_seq)

        aligned_sequences = aligned_sequences_new
        for seq in aligned_sequences:
            if len(seq) == len(first_seq):
                count += 1

    aligned_sequences.insert(0, first_seq)
    return aligned_sequences


# Example usage
ocr_outputs = [
    "N ăm ấy tôi mười hại tuổi. Tôi ở thị trấn Rêu với haha",
    "Năm ấy tôi mơi hai tuổi. Tôi thị trấn Rêu với",
    "N ăm ấy tôi mười hai tuồi. Tôi ở thị trấn Reu voi",
    "Na ay toi mười hai tuổi Tôi o thi tr Rêu với"
]

aligned_result = align_multiple_sequences(ocr_outputs)
for seq in aligned_result:
    print(f'"{seq}",')
