import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import Predict from './pages/Predict';
import Monitoring from './pages/Monitoring';
import History from './pages/History';

export const API_URL = 'http://localhost:8000';

const NavItem = ({ to, icon, label, active }) => (
  <Link to={to} className={`nav-item ${active ? 'active' : ''}`} style={{ textDecoration: 'none' }}>
    <span className="nav-icon">{icon}</span>
    <span>{label}</span>
  </Link>
);

const AppContent = () => {
  const location = useLocation();

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="logo-icon">🛡️</div>
          <div>
            <div className="logo-text">ChurnGuard</div>
            <div className="logo-sub">MLOps Analytics</div>
          </div>
        </div>
        <nav className="sidebar-nav">
          <div className="nav-section-label">Operations</div>
          <NavItem
            to="/"
            icon="🚀"
            label="Predictions"
            active={location.pathname === '/'}
          />
          <NavItem
            to="/history"
            icon="📋"
            label="History"
            active={location.pathname === '/history'}
          />

          <div className="nav-section-label">Monitoring</div>
          <NavItem
            to="/monitoring"
            icon="📊"
            label="Drift Analysis"
            active={location.pathname === '/monitoring'}
          />
        </nav>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <h1 className="topbar-title">
              {location.pathname === '/' && 'Predict Customer Churn'}
              {location.pathname === '/history' && 'Prediction History'}
              {location.pathname === '/monitoring' && 'Model Monitoring'}
            </h1>
            <p className="topbar-sub">Real-time telco churn analytics pipeline</p>
          </div>
          <div className="status-badge">
            <div className="status-dot"></div>
            <span>System Online</span>
          </div>
        </header>

        <div className="page-content">
          <Routes>
            <Route path="/" element={<Predict />} />
            <Route path="/monitoring" element={<Monitoring />} />
            <Route path="/history" element={<History />} />
          </Routes>
        </div>
      </main>
    </div>
  );
};

const App = () => (
  <Router>
    <AppContent />
  </Router>
);

export default App;
