import { useState, useEffect } from 'react'
import axios from 'axios'

export default function TransactionAnalyzer() {
  const [features, setFeatures] = useState([])
  const [values, setValues] = useState(Array(31).fill(0))
  const [result, setResult] = useState(null)
  const [explanation, setExplanation] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingExplain, setLoadingExplain] = useState(false)

  useEffect(() => {
    axios.get('/api/feature-names').then(r => setFeatures(r.data.features))
  }, [])

  const loadSample = async (type) => {
    const r = await axios.get('/api/sample-transactions')
    setValues(r.data[type].values)
    setResult(null)
    setExplanation(null)
  }

  const predict = async () => {
    setLoading(true)
    setExplanation(null)
    try {
      const r = await axios.post('/api/predict', { features: values.map(Number) })
      setResult(r.data)
    } catch (e) {
      setResult({ error: e.response?.data?.error || e.message })
    }
    setLoading(false)
  }

  const explain = async () => {
    setLoadingExplain(true)
    try {
      const r = await axios.post('/api/explain', { features: values.map(Number) })
      setExplanation(r.data)
    } catch (e) {
      setExplanation({ error: e.response?.data?.error || e.message })
    }
    setLoadingExplain(false)
  }

  const prob = result?.fraud_probability ?? 0
  const gaugeColor = prob > 0.85 ? '#ef4444' : prob > 0.5 ? '#f59e0b' : '#10b981'
  const gaugeGrad = `conic-gradient(${gaugeColor} ${prob * 360}deg, var(--border) 0deg)`

  return (
    <div>
      <div className="page-header">
        <h2>🔍 Transaction Analyzer</h2>
        <p>Input transaction features and get real-time fraud predictions with explanations</p>
      </div>

      {/* Quick Actions */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header"><h3>Quick Actions</h3></div>
        <div className="btn-group">
          <button className="btn btn-secondary" onClick={() => loadSample('legitimate')}>
            ✅ Load Legitimate Sample
          </button>
          <button className="btn btn-secondary" onClick={() => loadSample('fraudulent')}>
            🚨 Load Fraud Sample
          </button>
          <button className="btn btn-secondary" onClick={() => { setValues(Array(31).fill(0)); setResult(null); setExplanation(null) }}>
            🔄 Reset All
          </button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: 20 }}>
        {/* Feature Input */}
        <div className="card">
          <div className="card-header"><h3>Transaction Features ({features.length})</h3></div>
          <div className="feature-grid">
            {features.map((f, i) => (
              <div className="form-group" key={f.name}>
                <label>
                  {f.protected && '🔒 '}{f.name}
                  <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}> — {f.description}</span>
                </label>
                <input
                  type="number" step="0.1"
                  value={values[i]}
                  onChange={e => { const v = [...values]; v[i] = parseFloat(e.target.value) || 0; setValues(v) }}
                />
              </div>
            ))}
          </div>
          <div style={{ marginTop: 16 }}>
            <button className="btn btn-primary" onClick={predict} disabled={loading}>
              {loading ? <><span className="spinner"></span> Predicting...</> : '🔍 Analyze Transaction'}
            </button>
          </div>
        </div>

        {/* Results Panel */}
        <div>
          {result && !result.error && (
            <div className="card" style={{ marginBottom: 20 }}>
              <div className="card-header"><h3>Prediction Result</h3></div>
              <div className="gauge-container">
                <div className="gauge-ring" style={{ background: gaugeGrad }}>
                  <span className="gauge-value" style={{ color: gaugeColor }}>
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="gauge-label" style={{ color: gaugeColor }}>
                  {result.is_fraud ? '🚨 FRAUD DETECTED' : '✅ LEGITIMATE'}
                </div>
                <div className="gauge-sublabel">
                  Threshold: {(result.threshold * 100).toFixed(1)}% · Confidence: {result.confidence}
                </div>
              </div>

              {result.is_fraud && (
                <button className="btn btn-primary" onClick={explain} disabled={loadingExplain} style={{ width: '100%' }}>
                  {loadingExplain ? <><span className="spinner"></span> Generating...</> : '🔒 Generate Explanation'}
                </button>
              )}
            </div>
          )}

          {result?.error && (
            <div className="card" style={{ borderColor: 'var(--danger)', marginBottom: 20 }}>
              <p style={{ color: 'var(--danger)' }}>❌ {result.error}</p>
            </div>
          )}

          {/* Explanation Results */}
          {explanation && !explanation.error && (
            <>
              {/* Protected Attributes */}
              <div className="card" style={{ marginBottom: 20 }}>
                <div className="card-header"><h3>🔒 Protected Attributes</h3></div>
                <div className="protected-list">
                  {Object.entries(explanation.protected_attributes || {}).map(([k, v]) => (
                    <span className="protected-tag" key={k}>🔒 {k}: {v.value}</span>
                  ))}
                </div>
              </div>

              {/* Counterfactuals */}
              {explanation.counterfactuals?.length > 0 && (
                <div className="card" style={{ marginBottom: 20 }}>
                  <div className="card-header"><h3>📋 Actionable Recourse</h3></div>
                  {explanation.counterfactuals.map(cf => (
                    <div className="cf-option" key={cf.id}>
                      <h4>Option {cf.id} → {(cf.new_probability * 100).toFixed(1)}% fraud probability</h4>
                      {Object.entries(cf.changes || {}).map(([feat, ch]) => (
                        <div className="change-item" key={feat}>
                          <span>{feat}</span>
                          <span><span style={{ color: 'var(--danger)' }}>{ch.from}</span> <span className="change-arrow">→</span> <span style={{ color: 'var(--success)' }}>{ch.to}</span></span>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              )}

              {/* Privacy Validation */}
              {explanation.privacy_validation?.length > 0 && (
                <div className="card">
                  <div className="card-header"><h3>🔐 Privacy Validation</h3></div>
                  {explanation.privacy_validation.map(pv => (
                    <div key={pv.counterfactual_id} style={{ padding: '6px 0', fontSize: '0.85rem' }}>
                      <span className={`badge ${pv.privacy_preserved ? 'badge-success' : 'badge-danger'}`}>
                        {pv.status}
                      </span>
                      <span style={{ marginLeft: 8, color: 'var(--text-secondary)' }}>Option {pv.counterfactual_id}</span>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
