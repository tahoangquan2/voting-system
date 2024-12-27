import google.generativeai as genai
import os
from datetime import datetime

# Configure the API key
GOOGLE_API_KEY = os.environ.get('dummy')
genai.configure(api_key='dummy')

# Choose a Gemini model
model = genai.GenerativeModel('gemini-exp-1206')

def correct_sentence(sentence):
    try:
        prompt = f"Correct Vietnamese text, output correct text only: '{sentence}'"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

def process_sentences_from_file(filepath):
    # Create output filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    corrected_file = f"corrected_{timestamp}.txt"
    log_file = f"processing_log_{timestamp}.txt"

    try:
        with open(filepath, 'r', encoding='utf-8') as f_in, \
             open(corrected_file, 'w', encoding='utf-8') as f_corrected, \
             open(log_file, 'w', encoding='utf-8') as f_log:

            # Write initial log entry
            f_log.write(f"Processing started at: {datetime.now()}\n")
            f_log.write(f"Input file: {filepath}\n\n")

            for line_num, line in enumerate(f_in, 1):
                sentence = line.strip()
                if sentence:  # Skip empty lines
                    corrected = correct_sentence(sentence)

                    f_corrected.write(f"{corrected}")

                    # Write detailed information to log file
                    f_log.write(f"Line {line_num}:\n")
                    f_log.write(f"Original: {sentence}\n")
                    f_log.write(f"Corrected: {corrected}\n")
                    f_log.write("-" * 50 + "\n")

                    print(f"Processed line {line_num}: {sentence}")

            f_log.write(f"\nProcessing completed at: {datetime.now()}")

    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        with open(log_file, 'w', encoding='utf-8') as f_log:
            f_log.write(f"ERROR: File not found: {filepath}\n")
    except Exception as e:
        print(f"Error occurred: {e}")
        with open(log_file, 'a', encoding='utf-8') as f_log:
            f_log.write(f"\nERROR: {e}\n")

if __name__ == "__main__":
    input_file = "sentences.txt"
    process_sentences_from_file(input_file)
