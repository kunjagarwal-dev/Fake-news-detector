import { useState } from "react";
import { predictNews } from "../../api/api";

export default function PredictionForm({ onResult }) {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (text.trim().length < 10) {
      setError("Enter at least 10 characters of article text.");
      return;
    }

    setLoading(true);
    try {
      const result = await predictNews(text);
      onResult(result, text);
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form className="prediction-form" onSubmit={handleSubmit}>
      <label htmlFor="article-text" className="form-label">
        Paste a news article
      </label>
      <textarea
        id="article-text"
        className="article-input"
        rows={10}
        placeholder="Paste the full article text here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <div className="form-footer">
        <span className="char-count">{text.length} characters</span>
        <button type="submit" className="submit-btn" disabled={loading}>
          {loading ? "Analyzing..." : "Check article"}
        </button>
      </div>
      {error && <p className="form-error">{error}</p>}
    </form>
  );
}
