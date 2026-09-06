import { useState } from "react";
import PredictionForm from "./components/PredictionForm";
import ResultCard from "./components/ResultCard";
import "./App.css";

export default function App() {
  const [result, setResult] = useState(null);

  const handleResult = (data) => {
    setResult(data);
  };

  return (
    <div className="app">
      <div className="app-container">
        <header className="app-header">
          <h1 className="app-title">Fake News Detector</h1>
          <p className="app-subtitle">
            Paste an article below to check whether it's likely real or fake,
            powered by a machine learning model trained on 38,000+ news
            articles.
          </p>
        </header>

        <main className="app-main">
          <PredictionForm onResult={handleResult} />
          <ResultCard result={result} />
        </main>

        <footer className="app-footer">
          <p>Built with FastAPI + scikit-learn + React</p>
        </footer>
      </div>
    </div>
  );
}
