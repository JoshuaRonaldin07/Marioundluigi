"""
Emotions Dataset Normaliser (for ISEAR)
-------------------------
This script processes a CSV file containing one emotion label and one text per row.
It cleans, normalises, and sorts the rows.
Ultimately produces a new CSV we can actually use (in the working directory).
"""

import csv
import os
import re

def process_emotion_data():
    # Step 1: Get the path to the CSV file
    file_path = input("Enter the path to the CSV file: ")
    
    # Extract the filename without the extension for later use
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    
    try:
        # Step 2: Read the CSV file using the csv module to correctly handle quoted fields
        rows = []
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for parts in csv_reader:
                if len(parts) < 2:  # Skip lines that don't have at least a label and some text
                    continue
                emotion = parts[0]
                # Combine the title (parts[1]) and review text (parts[2], etc.) into one string
                text = " ".join(parts[1:])
                rows.append([emotion, text])
        
        # Step 3: Remove rows with at least one of the two columns empty
        rows = [row for row in rows if row[0] and row[1].strip()]
        
        # Step 4: Sort rows alphabetically based on the first column (emotions)
        rows.sort(key=lambda x: x[0].lower())
        
        # Step 5: Normalise the texts in the second column
        for i in range(len(rows)):
            # Convert to lowercase and remove anything that's not a letter or a space
            normalised_text = rows[i][1].lower()
            normalised_text = re.sub(r'[^a-z ]', '', normalised_text)
            rows[i][1] = normalised_text
        
        # Step 6: Report the total number of finalised rows and display a number of lines as sample if wanted
        print(f"Total number of processed rows: {len(rows)}")
        rows_to_display = input("Enter the number of rows to display (0 to skip): ")
        
        try:
            rows_to_display = int(rows_to_display)
            if rows_to_display > 0:
                print("\nSample of processed data:")
                for i in range(min(rows_to_display, len(rows))):
                    print(f"{rows[i][0]}, {rows[i][1]}")
        except ValueError:
            print("Invalid input. Skipping display.")
        
        # Step 7: Output the processed array to a new CSV file
        output_filename = f"{base_filename}-n.csv"
        with open(output_filename, 'w', newline='', encoding='utf-8') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows(rows)
        
        print(f"\nProcessed data saved to {output_filename}")
    
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    process_emotion_data()