const API_URL =
  globalThis.process?.env?.REACT_APP_API_URL || "http://localhost:8000";

export async function predictNews(text) {
  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || "Something went wrong. Try again.");
  }

  return response.json();
}
