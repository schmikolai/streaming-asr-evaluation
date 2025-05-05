from src.eval.Dataset import Dataset
from src.helper.logger import init_logger

init_logger()

BYTES_PER_SAMPLE = 2
SAMPLE_RATE = 16000

d = Dataset()

total_bytes = 0

for el in d:
    total_bytes += len(el[1])

print(f"Total bytes in dataset: {total_bytes}")
print(f"Total seconds in dataset: {total_bytes / (BYTES_PER_SAMPLE * SAMPLE_RATE)}")
