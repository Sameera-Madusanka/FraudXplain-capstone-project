import { useState, useEffect } from 'react'
import axios from 'axios'

export default function Explainability() {
  const [features, setFeatures] = useState([])
  const [values, setValues] = useState(Array(31).fill(0))
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    axios.get('/api/feature-names').then(r => setFeatures(r.data.features))
  }, [])

  const loadSample = async (type) => {
    const r = await axios.get('/api/sample-transactions')
    setValues(r.data[type].values)
    setResult(null)
  }

  const generateExplanation = async () => {
    setLoading(true)
    try {
      const r = await axios.post('/api/explain', { features: values.map(Number) })
      setResult(r.data)
    } catch (e) {
      setResult({ error: e.response?.data?.error || e.message })
    }
    setLoading(false)
  }

  const prob = result?.fraud_probability ?? 0
  const gaugeColor = prob > 0.85 ? '#ef4444' : prob > 0.5 ? '#f59e0b' : '#10b981'
  const gaugeGrad = `conic-gradient(${gaugeColor} ${prob * 360}deg, var(--border) 0deg)`

  return (
    <div>
      <div className="page-header">
        <h2>🔒 Constrained Counterfactual Explanations</h2>
        <p>Privacy-preserving explanations that protect sensitive attributes while providing actionable recourse</p>
      </div>

      {/* Research Context Banner */}
      <div className="card" style={{ marginBottom: 20, borderColor: 'var(--accent)', background: 'var(--accent-glow)' }}>
        <h3 style={{ fontSize: '0.95rem', marginBottom: 8 }}>📖 Research Contribution</h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.7 }}>
          This system generates <strong style={{ color: 'var(--accent-hover)' }}>constrained counterfactual explanations</strong> that
          answer: <em>"What actionable changes could make this transaction non-fraudulent?"</em> — while
          <strong style={{ color: 'var(--success)' }}> formally guaranteeing</strong> that protected attributes
          (income, age, employment status, housing) are <strong>never</strong> modified or suggested for change.
          This is the novel contribution combining <strong>Federated Learning</strong> with <strong>privacy-preserving explainability</strong>.
        </p>
      </div>

      {/* How It Works */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header"><h3>How Constrained Counterfactuals Work</h3></div>
        <div className="arch-flow">
          {[
            { title: '1. Classify', detail: 'Model predicts\nfraud probability' },
            { title: '2. Constrain', detail: 'Lock protected\nattributes' },
            { title: '3. Optimize', detail: 'Perturb only\nactionable features' },
            { title: '4. Validate', detail: 'Verify privacy\npreservation' },
            { title: '5. Explain', detail: 'Actionable\nrecourse options' },
          ].map((b, i, arr) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div className={`arch-block ${i === 1 || i === 3 ? 'highlight' : ''}`}>
                <div className="block-title">{b.title}</div>
                <div className="block-detail" style={{ whiteSpace: 'pre-line' }}>{b.detail}</div>
              </div>
              {i < arr.length - 1 && <span className="arch-arrow">→</span>}
            </div>
          ))}
        </div>
      </div>

      {/* Quick Load + Generate */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header"><h3>Test Transaction</h3></div>
        <div className="btn-group" style={{ marginBottom: 16 }}>
          <button className="btn btn-secondary" onClick={() => loadSample('legitimate')}>✅ Load Legitimate</button>
          <button className="btn btn-secondary" onClick={() => loadSample('fraudulent')}>🚨 Load Fraudulent</button>
        </div>

        {/* Compact feature display */}
        <details style={{ marginBottom: 16 }}>
          <summary style={{ cursor: 'pointer', color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: 8 }}>
            📝 Edit Transaction Features ({features.length} features)
          </summary>
          <div className="feature-grid" style={{ marginTop: 12 }}>
            {features.map((f, i) => (
              <div className="form-group" key={f.name} style={{ marginBottom: 8 }}>
                <label>
                  {f.protected && '🔒 '}{f.name}
                </label>
                <input
                  type="number" step="0.1"
                  value={values[i]}
                  onChange={e => { const v = [...values]; v[i] = parseFloat(e.target.value) || 0; setValues(v) }}
                  style={f.protected ? { borderColor: 'rgba(16,185,129,0.3)', background: 'rgba(16,185,129,0.05)' } : {}}
                />
              </div>
            ))}
          </div>
        </details>

        <button className="btn btn-primary" onClick={generateExplanation} disabled={loading} style={{ width: '100%' }}>
          {loading ? <><span className="spinner"></span> Generating Constrained Counterfactuals...</> : '🔒 Generate Privacy-Preserving Explanation'}
        </button>
      </div>

      {result?.error && (
        <div className="card" style={{ borderColor: 'var(--danger)', marginBottom: 20 }}>
          <p style={{ color: 'var(--danger)' }}>❌ {result.error}</p>
        </div>
      )}

      {result && !result.error && (
        <>
          {/* Detection Result */}
          <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: 20, marginBottom: 20 }}>
            <div className="card">
              <div className="gauge-container">
                <div className="gauge-ring" style={{ background: gaugeGrad }}>
                  <span className="gauge-value" style={{ color: gaugeColor }}>{(prob * 100).toFixed(1)}%</span>
                </div>
                <div className="gauge-label" style={{ color: gaugeColor }}>
                  {result.is_fraud ? '🚨 FRAUD' : '✅ LEGITIMATE'}
                </div>
                <div className="gauge-sublabel">Threshold: {(result.threshold * 100).toFixed(1)}%</div>
              </div>
            </div>

            {/* Protected Attributes (KEY SECTION) */}
            <div className="card" style={{ borderColor: 'rgba(16,185,129,0.3)' }}>
              <div className="card-header">
                <h3>🔒 Protected Attributes — NEVER Changed</h3>
                <span className="badge badge-success">Privacy Guaranteed</span>
              </div>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: 12 }}>
                These sensitive attributes are formally constrained — counterfactuals will never suggest modifying them.
              </p>
              <table>
                <thead><tr><th>Attribute</th><th>Value</th><th>Description</th><th>Status</th></tr></thead>
                <tbody>
                  {Object.entries(result.protected_attributes || {}).map(([k, v]) => (
                    <tr key={k}>
                      <td style={{ fontWeight: 600 }}>🔒 {k}</td>
                      <td><code style={{ color: 'var(--success)' }}>{v.value}</code></td>
                      <td style={{ color: 'var(--text-muted)' }}>{v.description}</td>
                      <td><span className="badge badge-success">{v.status}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Counterfactual Explanations (CORE SECTION) */}
          {result.counterfactuals?.length > 0 && (
            <div className="card" style={{ marginBottom: 20 }}>
              <div className="card-header">
                <h3>📋 Constrained Counterfactual Explanations ({result.counterfactuals.length})</h3>
              </div>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: 16 }}>
                Each option shows the <strong>minimal changes to actionable attributes only</strong> that could change the prediction from Fraud to Legitimate.
                Protected attributes (income, age, employment, housing) are <strong>never</strong> included.
              </p>

              {result.counterfactuals.map(cf => (
                <div className="cf-option" key={cf.id} style={{ borderColor: cf.new_class === 'Legitimate' ? 'rgba(16,185,129,0.3)' : 'var(--border)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                    <h4 style={{ margin: 0 }}>
                      Counterfactual Option {cf.id}
                    </h4>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                      <span className={`badge ${cf.new_class === 'Legitimate' ? 'badge-success' : 'badge-warning'}`}>
                        → {cf.new_class} ({(cf.new_probability * 100).toFixed(1)}%)
                      </span>
                      {cf.privacy_validated && <span className="badge badge-success">🔒 Privacy OK</span>}
                    </div>
                  </div>

                  {cf.changes?.length > 0 ? (
                    <table>
                      <thead>
                        <tr><th>Feature</th><th>Original</th><th></th><th>Counterfactual</th><th>Change</th></tr>
                      </thead>
                      <tbody>
                        {cf.changes.map((ch, j) => (
                          <tr key={j}>
                            <td style={{ fontWeight: 600 }}>
                              {ch.is_actionable && '✏️ '}{ch.feature}
                              <br /><span style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontWeight: 400 }}>{ch.description}</span>
                            </td>
                            <td><code style={{ color: 'var(--danger)' }}>{ch.original}</code></td>
                            <td><span className="change-arrow">→</span></td>
                            <td><code style={{ color: 'var(--success)' }}>{ch.counterfactual}</code></td>
                            <td style={{ color: 'var(--text-muted)' }}>±{ch.change_magnitude}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No significant actionable changes identified.</p>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Privacy Validation Results (KEY SECTION) */}
          {result.privacy_validation?.length > 0 && (
            <div className="card" style={{ marginBottom: 20, borderColor: 'rgba(16,185,129,0.3)' }}>
              <div className="card-header">
                <h3>🔐 Privacy Validation Results</h3>
                <span className="badge badge-success">
                  {result.privacy_validation.filter(p => p.privacy_preserved).length}/{result.privacy_validation.length} Passed
                </span>
              </div>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: 12 }}>
                Each counterfactual is validated against privacy constraints to ensure no protected attributes were modified.
              </p>
              {result.privacy_validation.map(pv => (
                <div key={pv.counterfactual_id} style={{
                  padding: '12px 16px',
                  background: pv.privacy_preserved ? 'var(--success-bg)' : 'var(--danger-bg)',
                  borderRadius: 'var(--radius-sm)',
                  marginBottom: 8,
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center'
                }}>
                  <div>
                    <span style={{ fontWeight: 600, fontSize: '0.88rem' }}>Counterfactual #{pv.counterfactual_id}</span>
                    <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: '2px 0 0' }}>{pv.status}</p>
                  </div>
                  <span className={`badge ${pv.privacy_preserved ? 'badge-success' : 'badge-danger'}`} style={{ fontSize: '0.82rem' }}>
                    {pv.privacy_preserved ? '🔒 PASSED' : '❌ FAILED'}
                  </span>
                </div>
              ))}

              {/* Overall Privacy Report */}
              {result.privacy_report && (
                <div style={{ marginTop: 12, padding: 12, background: 'var(--bg-primary)', borderRadius: 'var(--radius-sm)' }}>
                  <p style={{ fontSize: '0.82rem', fontWeight: 600, marginBottom: 4 }}>Overall Privacy Report</p>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                    Total violations: {result.privacy_report.total_violations ?? 0} ·
                    Compliance: <span style={{ color: result.privacy_report.privacy_compliant ? 'var(--success)' : 'var(--danger)', fontWeight: 600 }}>
                      {result.privacy_report.privacy_compliant ? '✅ FULLY COMPLIANT' : '❌ NON-COMPLIANT'}
                    </span>
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Actionable Recourse Text */}
          {result.actionable_recourse && (
            <div className="card" style={{ marginBottom: 20 }}>
              <div className="card-header"><h3>📝 Actionable Recourse Report</h3></div>
              <pre style={{
                background: 'var(--bg-primary)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-sm)',
                padding: 16,
                fontSize: '0.82rem',
                color: 'var(--text-secondary)',
                whiteSpace: 'pre-wrap',
                lineHeight: 1.6,
                maxHeight: 400,
                overflow: 'auto'
              }}>
                {result.actionable_recourse}
              </pre>
            </div>
          )}

          {/* Not fraud message */}
          {!result.is_fraud && (
            <div className="card" style={{ borderColor: 'rgba(16,185,129,0.3)' }}>
              <div style={{ textAlign: 'center', padding: 24 }}>
                <p style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--success)', marginBottom: 8 }}>✅ Transaction is Legitimate</p>
                <p style={{ color: 'var(--text-muted)' }}>Counterfactual explanations are generated only for transactions classified as fraudulent.</p>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: 8 }}>Try loading a fraudulent sample to see the constrained counterfactual explanations.</p>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
