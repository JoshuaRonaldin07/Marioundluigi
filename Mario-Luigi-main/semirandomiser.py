"""
Emotions Dataset Semi-Randomiser (for isear)
-------------------------
This script processes a CSV file containing one emotion label and one text per row.
It randomises every other label to create a supposed "model output".
We will be using this file to develop and test our evaluation function(s).
"""

import csv
import os
import random

def process_emotion_data():
    # Step 1: Get the path to the CSV file
    file_path = input("Enter the path to the CSV file: ")
    
    # Extract the filename without the extension for later use
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    
    try:
        # Step 2: Read the CSV
        rows = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line and ',' in line:
                    # Split at the first comma
                    parts = line.split(',', 1)
                    emotion = parts[0].strip()
                    text = parts[1].strip() if len(parts) > 1 else ""
                    rows.append([emotion, text])
        
        print(f"Total number of rows read: {len(rows)}")
        
        # Step 3: Make a copy of the array to work with
        randomized_rows = [row.copy() for row in rows]
        
        # Step 4: Scramble the emotions while maintaining their counts
        # Extract the emotions
        even_emotions = [randomized_rows[i][0] for i in range(len(randomized_rows)) if i % 2 == 0]
        # Shuffle the emotions
        random.shuffle(even_emotions)
        # Replace the emotions in the randomised array
        even_index = 0
        for i in range(len(randomized_rows)):
            if i % 2 == 0:
                randomized_rows[i][0] = even_emotions[even_index]
                even_index +=1
        
        # Step 5: Compare the original and the randomised arrays
        similarity = 0
        total_rows = len(rows)
        
        for i in range(total_rows):
            if rows[i][0] == randomized_rows[i][0]:
                similarity += 1
        
        similarity_percentage = (similarity / total_rows) * 100 if total_rows > 0 else 0
        
        print(f"\nComparison Results:")
        print(f"Total rows: {total_rows}")
        print(f"Unchanged labels: {similarity}")
        print(f"Similarity: {similarity_percentage:.2f}%")
        
        # Setep 6: Output the randomised array to a new CSV file
        output_filename = f"{base_filename}_semi.csv"
        with open(output_filename, 'w', newline='', encoding='utf-8') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows(randomized_rows)
        
        print(f"\nRandomized data saved to {output_filename}")
    
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    process_emotion_data()