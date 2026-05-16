import React, { useState, useEffect } from 'react';

export default function Monitoring() {
  const [drift, setDrift] = useState(null);
  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  useEffect(() => {
    fetch(`${API_URL}/drift/latest`)
      .then(r => r.json())
      .then(setDrift)
      .catch(() => {});
  }, []);

  const tools = [
    { name: 'Evidently AI', desc: 'Data & target drift detection', icon: '🔍', status: 'Running', color: 'green', link: '#' },
    { name: 'MLflow Registry', desc: 'Experiment tracking', icon: '🏭', status: 'Active', color: 'green', link: 'http://localhost:5001' },
    { name: 'Prometheus', desc: 'Metrics & alerting', icon: '🔥', status: 'Scraping', color: 'green', link: 'http://localhost:9090' },
    { name: 'Grafana', desc: 'Dashboard visualization', icon: '📊', status: 'Ready', color: 'green', link: 'http://localhost:3002' },
    { name: 'Marquez', desc: 'Data lineage', icon: '🗺️', status: 'Tracking', color: 'blue', link: 'http://localhost:3000' },
    { name: 'ClickHouse', desc: 'Analytical warehouse', icon: '🏠', status: 'Connected', color: 'green', link: 'http://localhost:8123' },
    { name: 'Great Expectations', desc: 'Data quality', icon: '✅', status: 'Validated', color: 'green', link: '#' },
    { name: 'Feast', desc: 'Feature store', icon: '🍽️', status: 'Materialized', color: 'blue', link: '#' },
  ];

  return (
    <div>
      {drift && (
        <div style={{ background: drift.drift_detected ? 'rgba(239,68,68,0.1)' : 'rgba(16,185,129,0.1)', border: `1px solid ${drift.drift_detected ? 'rgba(239,68,68,0.3)' : 'rgba(16,185,129,0.3)'}`, borderRadius: 16, padding: '20px 24px', marginBottom: 24, display: 'flex', alignItems: 'center', gap: 16 }}>
          <div style={{ fontSize: 36 }}>{drift.drift_detected ? '🚨' : '✅'}</div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 18, color: drift.drift_detected ? '#ef4444' : '#10b981' }}>{drift.drift_detected ? 'Data Drift Detected!' : 'No Significant Data Drift'}</div>
            <div style={{ color: 'var(--text-muted)', fontSize: 13, marginTop: 4 }}>Drift score: {drift.drift_score} | Threshold: {drift.drift_threshold}</div>
          </div>
        </div>
      )}

      <div className="card" style={{ marginBottom: 24 }}>
        <div className="section-title"><span>⚙️</span> MLOps Stack Health</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 12 }}>
          {tools.map(t => (
            <div key={t.name} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '12px 16px', background: 'var(--bg-primary)', borderRadius: 10, border: '1px solid var(--border)' }}>
              <span style={{ fontSize: 24 }}>{t.icon}</span>
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 600, fontSize: 13 }}>{t.name}</div>
                <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>{t.desc}</div>
              </div>
              <span className={`badge badge-${t.color}`}>{t.status}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="section-title"><span>🔗</span> Quick Links</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
          {[
            { label: '📊 Grafana', url: 'http://localhost:3002' },
            { label: '🏭 MLflow', url: 'http://localhost:5001' },
            { label: '🗺️ Marquez', url: 'http://localhost:3000' },
            { label: '🔥 Prometheus', url: 'http://localhost:9090' },
            { label: '🚀 API Docs', url: 'http://localhost:8000/docs' },
            { label: '🏠 ClickHouse', url: 'http://localhost:8123/play' },
          ].map(l => (
            <a key={l.label} href={l.url} target="_blank" rel="noreferrer" className="btn btn-secondary" style={{ fontSize: 13 }}>{l.label} ↗</a>
          ))}
        </div>
      </div>
    </div>
  );
}
