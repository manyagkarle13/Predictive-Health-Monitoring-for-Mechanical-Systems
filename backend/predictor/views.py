import os
import joblib
import pandas as pd
from django.http import JsonResponse
from rest_framework.decorators import api_view

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@api_view(['POST'])
def predict_rul(request):
    try:
        dataset = request.data.get("dataset")
        engine_id = int(request.data.get("engine_id", 1))
        cycle = float(request.data.get("cycle"))

        model_path = os.path.abspath(os.path.join(
            BASE_DIR, "..", "ml", "model", f"rul_model_{dataset}.pkl"
        ))

        model = joblib.load(model_path)

        # feature names used by the model
        feature_columns = list(model.feature_names_in_)

        # load dataset to fetch actual sensor values for the engine & cycle
        data_file = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', f'train_{dataset}.txt'))
        df_data = pd.read_csv(data_file, sep=r"\s+", header=None, engine='python')
        df_data = df_data.dropna(axis=1, how='all')

        # expected columns in the data files
        cols = ["engine_id", "cycle"]
        for i in range(1, 4):
            cols.append(f"op_setting_{i}")
        for i in range(1, 22):
            cols.append(f"sensor_{i}")
        # if file has extra cols, trim; if fewer, raise
        if df_data.shape[1] < len(cols):
            raise ValueError(f"Data file {data_file} has {df_data.shape[1]} columns but expected {len(cols)}")
        if df_data.shape[1] > len(cols):
            df_data = df_data.iloc[:, : len(cols)]
        df_data.columns = cols

        # find the row matching engine_id and cycle
        matching = df_data[(df_data['engine_id'] == engine_id) & (df_data['cycle'] == cycle)]
        if matching.empty:
            raise ValueError(f"No data row found for engine_id={engine_id} cycle={cycle} in dataset {dataset}")
        row = matching.iloc[0]

        # build feature vector (model expects specific feature columns)
        row_features = row.drop(labels=['engine_id'])
        # ensure DataFrame with one row and columns ordered per feature_columns
        X = pd.DataFrame(columns=feature_columns)
        for col in feature_columns:
            if col in row_features.index:
                X.at[0, col] = row_features[col]
            else:
                X.at[0, col] = 0

        predicted_rul = model.predict(X)[0]

        if predicted_rul < 30:
            status = "Critical"
        elif predicted_rul < 80:
            status = "Warning"
        else:
            status = "Healthy"

        return JsonResponse({
            "predicted_rul": int(predicted_rul),
            "health_status": status
        })

    except Exception as e:
        return JsonResponse({"error": str(e)})