import os
import json

dataset_path = os.path.join("data", "librispeech-pc-test-other")

entries = os.listdir(dataset_path)

for id in entries:
    if not os.path.isdir(os.path.join(dataset_path, id)):
        continue
    transcript_json_path = os.path.join(dataset_path, id, f"{id}.json")
    with open(transcript_json_path, "r") as f:
        transcript_json = f.read()
        transcript = json.loads(transcript_json)["transcript"]
        with open(os.path.join(dataset_path, id, f"{id}.txt"), "w") as f:
            f.write(transcript)
