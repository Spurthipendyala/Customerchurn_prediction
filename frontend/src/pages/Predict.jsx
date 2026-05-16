import React, { useState } from 'react';
import { API_URL } from '../App';

const defaultForm = {
  customerID: 'CUST-' + Math.random().toString(36).substr(2, 6).toUpperCase(),
  tenure: 12, MonthlyCharges: 65.5, TotalCharges: 786.0,
  SeniorCitizen: 0, Partner: 0, Dependents: 0, PhoneService: 1,
  MultipleLines: 0, OnlineSecurity: 0, OnlineBackup: 1, DeviceProtection: 0,
  TechSupport: 0, StreamingTV: 0, StreamingMovies: 0, PaperlessBilling: 1,
  gender_Male: 0, InternetService_Fiber: 0, InternetService_No: 0,
  Contract_OneYear: 0, Contract_TwoYear: 0,
  PaymentMethod_CreditCard: 0, PaymentMethod_ElecCheck: 1, PaymentMethod_MailedCheck: 0,
};

export default function Predict() {
  const [form, setForm] = useState(defaultForm);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setForm(f => ({ ...f, [name]: type === 'checkbox' ? (checked ? 1 : 0) : parseFloat(value) || value }));
  };

  const handlePreset = (preset) => {
    const presets = {
      'high-risk': { ...defaultForm, tenure: 2, MonthlyCharges: 99.5, TotalCharges: 199, PhoneService: 1, InternetService_Fiber: 1, PaperlessBilling: 1, PaymentMethod_ElecCheck: 1 },
      'low-risk': { ...defaultForm, tenure: 60, MonthlyCharges: 45, TotalCharges: 2700, Contract_TwoYear: 1, PaymentMethod_CreditCard: 1, PaymentMethod_ElecCheck: 0, OnlineSecurity: 1, OnlineBackup: 1, InternetService_No: 0 },
      'new-customer': { ...defaultForm, tenure: 1, MonthlyCharges: 70.5, TotalCharges: 70.5 },
    };
    setForm(f => ({ ...presets[preset], customerID: f.customerID }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true); setError(''); setResult(null);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const probPercent = result ? Math.round(result.churn_probability * 100) : 0;
  const isChurn = result?.churn_prediction === 1;
  const probColor = probPercent < 30 ? '#10b981' : probPercent < 50 ? '#f59e0b' : probPercent < 70 ? '#f97316' : '#ef4444';

  const checkboxFeatures = [
    { name: 'PhoneService', label: '📞 Phone Service' },
    { name: 'MultipleLines', label: '📱 Multiple Lines' },
    { name: 'OnlineSecurity', label: '🔒 Online Security' },
    { name: 'OnlineBackup', label: '☁️ Online Backup' },
    { name: 'DeviceProtection', label: '🛡️ Device Protection' },
    { name: 'TechSupport', label: '🛠️ Tech Support' },
    { name: 'StreamingTV', label: '📺 Streaming TV' },
    { name: 'StreamingMovies', label: '🎬 Streaming Movies' },
    { name: 'PaperlessBilling', label: '📧 Paperless Billing' },
    { name: 'Partner', label: '💑 Has Partner' },
    { name: 'Dependents', label: '👨‍👩‍👧 Has Dependents' },
    { name: 'SeniorCitizen', label: '👴 Senior Citizen' },
  ];

  return (
    <div>
      {/* Quick Preset Buttons */}
      <div className="card" style={{ marginBottom: 24, display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
        <span style={{ fontSize: 13, color: 'var(--text-muted)', fontWeight: 600 }}>QUICK PRESETS:</span>
        {[
          { id: 'high-risk', label: '🚨 High Risk Customer', color: 'var(--accent-red)' },
          { id: 'low-risk', label: '✅ Low Risk Customer', color: 'var(--accent-green)' },
          { id: 'new-customer', label: '🆕 New Customer', color: 'var(--accent-blue)' },
        ].map(p => (
          <button key={p.id} className="btn btn-secondary" onClick={() => handlePreset(p.id)}
            style={{ fontSize: 13, padding: '8px 16px', borderColor: p.color, color: p.color }}>
            {p.label}
          </button>
        ))}
      </div>

      <div className="predict-grid">
        {/* Form */}
        <div>
          <form onSubmit={handleSubmit}>
            <div className="card" style={{ marginBottom: 20 }}>
              <div className="section-title"><span>👤</span> Customer Identity</div>
              <div className="form-group">
                <label className="form-label">Customer ID</label>
                <input className="form-input" name="customerID" value={form.customerID} onChange={handleChange} />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                <div className="form-group">
                  <label className="form-label">Gender</label>
                  <select className="form-select" name="gender_Male" value={form.gender_Male} onChange={handleChange}>
                    <option value={0}>Female</option>
                    <option value={1}>Male</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Tenure (months)</label>
                  <input className="form-input" type="number" name="tenure" value={form.tenure} onChange={handleChange} min="0" max="120" />
                </div>
              </div>
            </div>

            <div className="card" style={{ marginBottom: 20 }}>
              <div className="section-title"><span>💰</span> Billing & Contract</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                <div className="form-group">
                  <label className="form-label">Monthly Charges ($)</label>
                  <input className="form-input" type="number" step="0.01" name="MonthlyCharges" value={form.MonthlyCharges} onChange={handleChange} />
                </div>
                <div className="form-group">
                  <label className="form-label">Total Charges ($)</label>
                  <input className="form-input" type="number" step="0.01" name="TotalCharges" value={form.TotalCharges} onChange={handleChange} />
                </div>
                <div className="form-group">
                  <label className="form-label">Contract Type</label>
                  <select className="form-select" onChange={e => {
                    const v = e.target.value;
                    setForm(f => ({ ...f, Contract_OneYear: v === '1' ? 1 : 0, Contract_TwoYear: v === '2' ? 1 : 0 }));
                  }} value={form.Contract_TwoYear ? '2' : form.Contract_OneYear ? '1' : '0'}>
                    <option value="0">Month-to-month</option>
                    <option value="1">One year</option>
                    <option value="2">Two year</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Payment Method</label>
                  <select className="form-select" onChange={e => {
                    const v = e.target.value;
                    setForm(f => ({
                      ...f,
                      PaymentMethod_ElecCheck: v === 'elec' ? 1 : 0,
                      PaymentMethod_MailedCheck: v === 'mail' ? 1 : 0,
                      PaymentMethod_CreditCard: v === 'credit' ? 1 : 0,
                    }));
                  }} value={form.PaymentMethod_CreditCard ? 'credit' : form.PaymentMethod_MailedCheck ? 'mail' : form.PaymentMethod_ElecCheck ? 'elec' : 'bank'}>
                    <option value="bank">Bank Transfer (auto)</option>
                    <option value="elec">Electronic Check</option>
                    <option value="mail">Mailed Check</option>
                    <option value="credit">Credit Card (auto)</option>
                  </select>
                </div>
              </div>
            </div>

            <div className="card" style={{ marginBottom: 20 }}>
              <div className="section-title"><span>📶</span> Internet Service</div>
              <div style={{ display: 'flex', gap: 10, marginBottom: 16 }}>
                {[{ v: '0', l: 'DSL' }, { v: 'fiber', l: 'Fiber optic' }, { v: 'none', l: 'No Internet' }].map(o => (
                  <label key={o.v} style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, padding: '12px', background: 'var(--bg-primary)', border: '1px solid var(--border)', borderRadius: 10, cursor: 'pointer', fontSize: 14, fontWeight: 500, transition: 'all 0.2s',
                    ...(((o.v === 'fiber' && form.InternetService_Fiber) || (o.v === 'none' && form.InternetService_No) || (o.v === '0' && !form.InternetService_Fiber && !form.InternetService_No)) ? { borderColor: 'var(--accent-blue)', background: 'rgba(59,130,246,0.1)', color: 'var(--accent-blue)' } : {}) }}>
                    <input type="radio" name="internet" value={o.v} style={{ display: 'none' }} checked={o.v === 'fiber' ? !!form.InternetService_Fiber : o.v === 'none' ? !!form.InternetService_No : !form.InternetService_Fiber && !form.InternetService_No}
                      onChange={() => setForm(f => ({ ...f, InternetService_Fiber: o.v === 'fiber' ? 1 : 0, InternetService_No: o.v === 'none' ? 1 : 0, has_internet: o.v === 'none' ? 0 : 1 }))} />
                    {o.l}
                  </label>
                ))}
              </div>
            </div>

            <div className="card" style={{ marginBottom: 20 }}>
              <div className="section-title"><span>⚙️</span> Services & Features</div>
              <div className="checkbox-group">
                {checkboxFeatures.map(feat => (
                  <label key={feat.name} className="checkbox-item">
                    <input type="checkbox" name={feat.name} checked={!!form[feat.name]} onChange={handleChange} />
                    {feat.label}
                  </label>
                ))}
              </div>
            </div>

            {error && (
              <div style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 10, padding: '12px 16px', color: 'var(--accent-red)', marginBottom: 16 }}>
                ❌ {error} — Is the API running at {API_URL}?
              </div>
            )}

            <button className="btn btn-primary btn-lg" type="submit" disabled={loading} style={{ width: '100%', justifyContent: 'center' }}>
              {loading ? <><div className="spinner" /> Predicting...</> : '🎯 Predict Churn Probability'}
            </button>
          </form>
        </div>

        {/* Result Panel */}
        <div>
          {!result && !loading && (
            <div className="card" style={{ textAlign: 'center', padding: '80px 40px' }}>
              <div style={{ fontSize: 64, marginBottom: 20 }}>🎯</div>
              <div style={{ fontSize: 20, fontWeight: 700, marginBottom: 12 }}>Ready to Predict</div>
              <div style={{ color: 'var(--text-muted)', fontSize: 14, lineHeight: 1.6 }}>
                Fill in the customer details on the left and click <strong style={{ color: 'var(--accent-blue)' }}>Predict Churn Probability</strong> to run ML inference.<br /><br />
                The model will return a churn probability score, risk level, and latency metrics.
              </div>
            </div>
          )}

          {loading && (
            <div className="card" style={{ textAlign: 'center', padding: '80px 40px' }}>
              <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 20 }}>
                <div className="spinner" style={{ width: 48, height: 48, borderWidth: 4, borderTopColor: 'var(--accent-blue)' }} />
              </div>
              <div style={{ fontSize: 16, fontWeight: 600 }}>Running ML Inference...</div>
              <div style={{ color: 'var(--text-muted)', fontSize: 13, marginTop: 8 }}>Loading model from MLflow Registry</div>
            </div>
          )}

          {result && (
            <div className="result-card">
              <div className={`result-header ${isChurn ? 'churn' : 'no-churn'}`}>
                <div className="result-icon">{isChurn ? '🚨' : '✅'}</div>
                <div>
                  <div className="result-title" style={{ color: isChurn ? '#ef4444' : '#10b981' }}>
                    {isChurn ? 'CHURN RISK DETECTED' : 'CUSTOMER LIKELY TO STAY'}
                  </div>
                  <div className="result-subtitle">Customer ID: {result.customerID}</div>
                  <div className={`risk-badge risk-${result.risk_level}`}>
                    ⚠️ {result.risk_level} RISK
                  </div>
                </div>
              </div>

              <div className="prob-bar-wrap">
                <div className="prob-bar-label">
                  <span>Churn Probability</span>
                  <span style={{ fontWeight: 700, fontSize: 24, color: probColor }}>{probPercent}%</span>
                </div>
                <div className="prob-bar-bg">
                  <div className="prob-bar-fill" style={{ width: `${probPercent}%`, background: `linear-gradient(90deg, ${probColor}88, ${probColor})` }} />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                  <span>0% — No Churn</span>
                  <span>50% — Threshold</span>
                  <span>100% — Certain Churn</span>
                </div>
              </div>

              <div className="meta-grid">
                {[
                  { label: 'Prediction', value: result.churn_label },
                  { label: 'Latency', value: `${result.latency_ms}ms` },
                  { label: 'Model', value: result.model_name.split('_').slice(-1)[0] },
                  { label: 'Version', value: `v${result.model_version}` },
                  { label: 'Prediction ID', value: result.prediction_id?.substr(0, 8) + '...' },
                  { label: 'Logged to ClickHouse', value: '✅ Yes' },
                ].map(item => (
                  <div key={item.label} className="meta-item">
                    <div className="meta-value">{item.value}</div>
                    <div className="meta-label">{item.label}</div>
                  </div>
                ))}
              </div>

              {isChurn && (
                <div style={{ padding: '0 28px 28px' }}>
                  <div style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 12, padding: '16px 20px' }}>
                    <div style={{ fontWeight: 700, marginBottom: 8, color: '#ef4444' }}>💡 Recommended Actions</div>
                    <ul style={{ color: 'var(--text-secondary)', fontSize: 13, paddingLeft: 16, lineHeight: 1.8 }}>
                      <li>Offer loyalty discount or contract upgrade</li>
                      <li>Proactive customer success outreach</li>
                      <li>Review and address service pain points</li>
                      <li>Consider bundle upgrade offer</li>
                    </ul>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
