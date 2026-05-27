import json

INPUT_FILE = "data.jsonl"
OUTPUT_FILE = "train.txt"

with open(INPUT_FILE) as f, \
     open(OUTPUT_FILE, "w") as out:
         
    for line in f:
        data = json.loads(line)
        text = data["text"].strip()
        out.write(text + "\n")

print("Done.")
