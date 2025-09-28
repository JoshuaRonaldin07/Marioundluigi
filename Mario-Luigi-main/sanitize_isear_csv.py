import csv
import os

def sanitize_csv(input_path, output_path):
    fixed_rows = []
    with open(input_path, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        for row in reader:
            if len(row) >= 2:
                label = row[0].strip()
                text = ','.join(row[1:]).strip()
                fixed_rows.append([label, text])
            else:
                print(f"Skipping malformed row: {row}")

    with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
        writer.writerows(fixed_rows)
    
    print(f"Sanitized file saved to: {output_path}")

if __name__ == "__main__":
    for split in ['train', 'val', 'test']:
        input_path = f"C:/Users/bluem/Desktop/Mario-Luigi-main/emotions/isear/isear-val.csv"
        output_path = f"C:/Users/bluem/Desktop/Mario-Luigi-main/emotions/isear/isear-val-sanitized.csv"
        sanitize_csv(input_path, output_path)
