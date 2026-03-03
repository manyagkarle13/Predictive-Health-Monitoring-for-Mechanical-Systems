import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [dataset, setDataset] = useState("FD001");
  const [engineId, setEngineId] = useState(1);
  const [cycle, setCycle] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  const handlePredict = async () => {
    try {
      setErrorMsg(null);
      setLoading(true);

      const response = await axios.post(
        "http://127.0.0.1:8000/api/predict/",
        {
          dataset: dataset,
          engine_id: parseInt(engineId),
          cycle: parseInt(cycle),
        }
      );

      setResult(response.data);
    } catch (error) {
      console.error(error);
      const msg = error?.response?.data?.error || error.message || "Prediction failed";
      setErrorMsg(msg);
    } finally {
      setLoading(false);
    }
  };

  const resultClass =
    result?.health_status === "Critical"
      ? "status-critical"
      : result?.health_status === "Warning"
      ? "status-warning"
      : "status-healthy";

  return (
    <div className="app-root">
      <div className="container">
        <div className="title">Predictive Health Monitoring Dashboard</div>

        <div className="card">
          <div className="form-row">
            <div className="label">Select Dataset</div>
            <select className="select" value={dataset} onChange={(e) => setDataset(e.target.value)}>
              <option value="FD001">FD001</option>
              <option value="FD002">FD002</option>
              <option value="FD003">FD003</option>
              <option value="FD004">FD004</option>
            </select>

            <div className="label">Enter Engine ID</div>
            <input className="input" type="number" value={engineId} onChange={(e) => setEngineId(e.target.value)} />

            <div className="label">Enter Engine Cycle</div>
            <input className="input" type="number" value={cycle} onChange={(e) => setCycle(e.target.value)} />

            <button className="btn" onClick={handlePredict} disabled={loading}>
              {loading ? "Predicting..." : "Predict RUL"}
            </button>

            {errorMsg && <div className="error">Error: {errorMsg}</div>}
          </div>
        </div>

        {result && result.predicted_rul && (
          <div className={`card result ${resultClass}`}>
            <h2>{result.predicted_rul} Cycles Remaining</h2>
            <h3>Status: {result.health_status}</h3>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
