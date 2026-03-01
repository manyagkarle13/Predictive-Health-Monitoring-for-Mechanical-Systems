from pathlib import Path
import pandas as pd
import joblib

# Paths (make robust regardless of current working directory)
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

# Load saved model from ml/model (relative to this script)
model_path = THIS_DIR / "model" / "rul_model.pkl"
features_path = THIS_DIR / "model" / "feature_columns.pkl"

if not model_path.exists() or not features_path.exists():
    raise FileNotFoundError(f"Model files not found in {THIS_DIR / 'model'}")

model = joblib.load(model_path)
feature_columns = joblib.load(features_path)

# Ask user for dataset
dataset_name = input("Enter dataset name (FD001/FD002/FD003/FD004): ").strip()
data_file = PROJECT_ROOT / "data" / f"train_{dataset_name}.txt"

if not data_file.exists():
    available = sorted(p.name for p in (PROJECT_ROOT / "data").glob('train_*.txt'))
    raise FileNotFoundError(
        f"Data file {data_file} not found. Available: {available}"
    )

# Load data (use whitespace delimiter to avoid extra empty columns)
df = pd.read_csv(data_file, sep=r"\s+", header=None, engine="python")
# Drop columns that are completely empty (some files have trailing spaces)
df = df.dropna(axis=1, how='all')
# Expecting 26 columns: engine_id, cycle, 3 op settings, 21 sensors
EXPECTED_COLS = 26
if df.shape[1] < EXPECTED_COLS:
    first_line = data_file.read_text(encoding='utf-8').splitlines()[0] if data_file.exists() else ''
    tokens = first_line.strip().split()
    raise ValueError(
        f"Parsed {df.shape[1]} columns but expected {EXPECTED_COLS}. "
        f"First line has {len(tokens)} tokens. Example tokens: {tokens[:40]}"
    )
if df.shape[1] > EXPECTED_COLS:
    df = df.iloc[:, :EXPECTED_COLS]

# Add column names
columns = ["engine_id", "cycle"]
for i in range(1, 4):
    columns.append(f"op_setting_{i}")
for i in range(1, 22):
    columns.append(f"sensor_{i}")
df.columns = columns

# Drop engine_id (keep features only)
features = df.drop(columns=["engine_id"])

# Select columns saved during training
features = features[feature_columns]

# Predict
predictions = model.predict(features)

def health_status(rul):
    if rul > 100:
        return "Healthy"
    elif rul > 50:
        return "Warning"
    else:
        return "Critical"

for i, value in enumerate(predictions[:800]):
    status = health_status(value)
    print(f"Cycle {i+1} → RUL: {int(value)} → Status: {status}")
