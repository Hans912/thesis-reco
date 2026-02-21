const API_BASE = '';

export async function sendChatMessage(messages, imageBase64 = null) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages: messages.map(m => ({ role: m.role, content: m.content })),
      image: imageBase64,
    }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Chat request failed: ${res.status} ${text}`);
  }
  return res.json();
}

export async function fetchProducts({ merchant, offset = 0, limit = 20 } = {}) {
  const params = new URLSearchParams({ offset, limit });
  if (merchant) params.set('merchant', merchant);
  const res = await fetch(`${API_BASE}/api/products?${params}`);
  if (!res.ok) throw new Error(`Failed to fetch products: ${res.status}`);
  return res.json();
}

export async function searchByText(query, { topK = 10, merchant } = {}) {
  const params = new URLSearchParams({ q: query, top_k: topK });
  if (merchant) params.set('merchant', merchant);
  const res = await fetch(`${API_BASE}/api/search/text?${params}`);
  if (!res.ok) throw new Error(`Search failed: ${res.status}`);
  return res.json();
}

export async function fetchStores({ merchant } = {}) {
  const params = new URLSearchParams();
  if (merchant) params.set('merchant', merchant);
  const res = await fetch(`${API_BASE}/api/stores?${params}`);
  if (!res.ok) throw new Error(`Failed to fetch stores: ${res.status}`);
  return res.json();
}

export async function fetchEvaluationResults(k = 5) {
  const res = await fetch(`${API_BASE}/api/evaluation/results?k=${k}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Evaluation failed: ${res.status} ${text}`);
  }
  return res.json();
}
