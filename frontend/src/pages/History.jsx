import React, { useState, useEffect } from 'react';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const RISK_COLORS = { LOW: 'green', MEDIUM: 'yellow', HIGH: 'yellow', CRITICAL: 'red' };

export default function History() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_URL}/predictions/history?limit=100`);
      if (!response.ok) throw new Error('Failed to fetch history');
      const data = await response.json();
      setHistory(data);
    } catch (err) {
      console.error('Error loading history:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadHistory(); }, []);

  const totalChurn = history.filter(h => h.churn_prediction === 1).length;
  const avgProb = history.length ? (history.reduce((a, b) => a + b.churn_probability, 0) / history.length * 100).toFixed(1) : 0;
  const avgLatency = history.length ? (history.reduce((a, b) => a + b.latency_ms, 0) / history.length).toFixed(1) : 0;

  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 16, marginBottom: 24 }}>
        {[
          { label: 'Total Predictions', value: history.length, icon: '🎯', color: 'blue' },
          { label: 'Churn Predicted', value: totalChurn, icon: '🚨', color: 'red' },
          { label: 'Avg Probability', value: `${avgProb}%`, icon: '📊', color: 'purple' },
          { label: 'Avg Latency', value: `${avgLatency}ms`, icon: '⚡', color: 'green' },
        ].map(s => (
          <div key={s.label} className={`stat-card ${s.color}`}>
            <div className={`stat-icon ${s.color}`}>{s.icon}</div>
            <div className="stat-value">{s.value}</div>
            <div className="stat-label">{s.label}</div>
          </div>
        ))}
      </div>

      <div className="card">
        <div className="section-title" style={{ justifyContent: 'space-between' }}>
          <span><span>🕐</span> Prediction History (ClickHouse)</span>
          <button className="btn btn-secondary" onClick={loadHistory} disabled={loading} style={{ fontSize: 12, padding: '6px 14px' }}>
            {loading ? '⌛ Loading...' : '🔄 Refresh'}
          </button>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>ID</th><th>Customer ID</th><th>Probability</th><th>Prediction</th>
                <th>Risk</th><th>Model</th><th>Latency</th><th>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {history.map(row => (
                <tr key={row.prediction_id}>
                  <td style={{ fontFamily: 'monospace', fontSize: 12, color: 'var(--text-muted)' }}>{row.prediction_id.substr(0,8)}</td>
                  <td style={{ fontWeight: 600 }}>{row.customerID}</td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{ width: 60, height: 6, background: 'var(--bg-primary)', borderRadius: 3, overflow: 'hidden' }}>
                        <div style={{ width: `${row.churn_probability * 100}%`, height: '100%', background: row.churn_probability > 0.5 ? '#ef4444' : '#10b981', borderRadius: 3 }} />
                      </div>
                      <span style={{ fontWeight: 700, fontSize: 13 }}>{(row.churn_probability * 100).toFixed(1)}%</span>
                    </div>
                  </td>
                  <td><span className={`badge ${row.churn_prediction === 1 ? 'badge-red' : 'badge-green'}`}>{row.churn_prediction === 1 ? 'CHURN' : 'NO CHURN'}</span></td>
                  <td><span className={`badge badge-${RISK_COLORS[row.risk_level] || 'blue'}`}>{row.risk_level}</span></td>
                  <td style={{ fontSize: 12, color: 'var(--text-muted)' }}>v{row.model_version}</td>
                  <td style={{ fontSize: 12 }}>{row.latency_ms}ms</td>
                  <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>{new Date(row.request_timestamp).toLocaleTimeString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ padding: '12px 0 0', fontSize: 12, color: 'var(--text-muted)' }}>
          📦 All predictions are stored in <code style={{ color: 'var(--accent-blue)' }}>churn_mlops.churn_predictions</code> ClickHouse table
        </div>
      </div>
    </div>
  );
}
