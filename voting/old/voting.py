from collections import Counter


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


aligned_outputs = [
    "N ăm ấy tôi mười hai tuồi. Tôi ở thị trấn Reu voi",
    "N`ăm ấy tôi m`ơi hai tuổi. Tôi ``thị trấn Rêu với",
    "N``a ay toi mười hai tuổi` Tôi o thi tr`` Rêu với",
    "N ăm ấy tôi mười hại tuổi. Tôi ở thị trấn Rêu với",
]

voted_output = vote_characters(aligned_outputs)
print(voted_output)
final_ouput = refine_output(voted_output)
print(final_ouput)
