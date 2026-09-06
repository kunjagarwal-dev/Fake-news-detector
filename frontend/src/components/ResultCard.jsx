export default function ResultCard({ result }) {
  if (!result) return null;

  const isReal = result.prediction === "real";
  const confidencePct = Math.round(result.confidence * 100);

  return (
    <div className={`result-card ${isReal ? "result-real" : "result-fake"}`}>
      <div className="result-icon">{isReal ? "✓" : "✕"}</div>
      <div className="result-body">
        <h3 className="result-label">
          {isReal ? "Likely real" : "Likely fake"}
        </h3>
        <p className="result-confidence">{confidencePct}% confidence</p>
        <div className="confidence-bar">
          <div
            className="confidence-fill"
            style={{ width: `${confidencePct}%` }}
          />
        </div>
      </div>
    </div>
  );
}
