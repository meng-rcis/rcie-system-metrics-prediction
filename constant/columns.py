COLS = [
    "time",
    "timestamp",
    "cpu_usage",
    "memory_usage",
    "bandwidth_inbound",
    "bandwidth_outbound",
    "tps",
    "tps_error",
    "response_time",
    "status",
]

FEATURES = [
    "cpu_usage",
    "memory_usage",
    "bandwidth_inbound",
    "bandwidth_outbound",
    "tps",
    "response_time",
]

LABEL = ["status"]

INDEX_COL = "time"

FREQUENCY = "5S"

# Window size for sliding window in classification
SIZE, TARGET_Y, MODE = 16, 8, 1
