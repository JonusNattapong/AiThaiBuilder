import csv
import glob
import os

output_file = "DataOutput/thai_sentiment_dataset.csv"
input_files = [
    "DataOutput/thai_happy_dataset.csv",
    "DataOutput/thai_sad_dataset.csv",
    "DataOutput/thai_angry_dataset.csv"
]

with open(output_file, "w", newline="", encoding="utf-8") as fout:
    writer = None
    for idx, file in enumerate(input_files):
        with open(file, "r", encoding="utf-8") as fin:
            reader = csv.reader(fin)
            header = next(reader)
            if writer is None:
                writer = csv.writer(fout)
                writer.writerow(header)
            for row in reader:
                writer.writerow(row)

print(f"Combined files into {output_file}")