import google.generativeai as genai
import os
import pandas as pd
from datetime import datetime
import time  # Add this import

# Configure the API key
GOOGLE_API_KEY = os.environ.get('AIzaSyDa2uW7tyne2Mrpap8zPczWsfsuJdXFSXY')
genai.configure(api_key='AIzaSyDa2uW7tyne2Mrpap8zPczWsfsuJdXFSXY')

# Choose a Gemini model
model = genai.GenerativeModel('gemini-exp-1206')

def correct_sentence(sentence):
    try:
        time.sleep(3.5)  # Add delay
        prompt = f"Correct Vietnamese text, it maybe not setence so dont auto uppercase first word, output correct text only: '{sentence}'"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

def process_csv_file(input_filepath):
    # Create output filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"corrected_{timestamp}.csv"
    log_file = f"processing_log_{timestamp}.txt"

    try:
        # Read the input CSV file
        df = pd.read_csv(input_filepath, encoding='utf-8', dtype=str)

        # Ensure the required columns exist
        if 'file' not in df.columns or 'text' not in df.columns:
            raise ValueError("CSV must contain 'file' and 'text' columns")

        # Create a new dataframe for the output
        output_df = df.copy()

        # Open log file
        with open(log_file, 'w', encoding='utf-8') as f_log:
            f_log.write(f"Processing started at: {datetime.now()}\n")
            f_log.write(f"Input file: {input_filepath}\n\n")

            # Process each row
            for index, row in df.iterrows():
                original_text = row['text']
                corrected_text = correct_sentence(original_text)
                output_df.at[index, 'text'] = corrected_text

                # Write to log file
                f_log.write(f"Entry {index + 1}:\n")
                f_log.write(f"File: {row['file']}\n")
                f_log.write(f"Original: {original_text}\n")
                f_log.write(f"Corrected: {corrected_text}\n")
                f_log.write("-" * 50 + "\n")

                print(f"Processed entry {index + 1}: {row['file']}")

            f_log.write(f"\nProcessing completed at: {datetime.now()}")

        # Save the output CSV
        output_df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\nProcessing complete. Output saved to {output_csv}")
        print(f"Log file saved to {log_file}")

    except FileNotFoundError:
        print(f"Error: File not found: {input_filepath}")
        with open(log_file, 'w', encoding='utf-8') as f_log:
            f_log.write(f"ERROR: File not found: {input_filepath}\n")
    except Exception as e:
        print(f"Error occurred: {e}")
        with open(log_file, 'a', encoding='utf-8') as f_log:
            f_log.write(f"\nERROR: {e}\n")

if __name__ == "__main__":
    input_file = "input.csv"
    process_csv_file(input_file)
