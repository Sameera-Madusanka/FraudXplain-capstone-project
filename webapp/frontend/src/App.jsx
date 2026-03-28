import { Routes, Route, NavLink } from 'react-router-dom'
import TransactionAnalyzer from './pages/TransactionAnalyzer'
import Explainability from './pages/Explainability'
import ModelDashboard from './pages/ModelDashboard'
import BatchTesting from './pages/BatchTesting'
import SystemArchitecture from './pages/SystemArchitecture'
import About from './pages/About'

function Sidebar() {
  const links = [
    { to: '/', icon: '🔍', label: 'Transaction Analyzer' },
    { to: '/explainability', icon: '🔒', label: 'Counterfactual Explanations' },
    { to: '/dashboard', icon: '📊', label: 'Model Dashboard' },
    { to: '/batch', icon: '📋', label: 'Batch Testing' },
    { to: '/architecture', icon: '🏗️', label: 'System Architecture' },
    { to: '/about', icon: 'ℹ️', label: 'About' },
  ]

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <h1>FraudXplain</h1>
        <span>Federated Fraud Detection</span>
      </div>
      <nav className="sidebar-nav">
        {links.map(l => (
          <NavLink key={l.to} to={l.to} end={l.to === '/'}>
            <span className="nav-icon">{l.icon}</span>
            <span>{l.label}</span>
          </NavLink>
        ))}
      </nav>
      <div className="sidebar-footer">
        FraudXplain v2.1 &middot; AUC 0.88
      </div>
    </aside>
  )
}

export default function App() {
  return (
    <div className="app-container">
      <Sidebar />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<TransactionAnalyzer />} />
          <Route path="/explainability" element={<Explainability />} />
          <Route path="/dashboard" element={<ModelDashboard />} />
          <Route path="/batch" element={<BatchTesting />} />
          <Route path="/architecture" element={<SystemArchitecture />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
    </div>
  )
}
