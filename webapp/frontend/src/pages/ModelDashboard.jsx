import { useState, useEffect } from 'react'
import axios from 'axios'

export default function ModelDashboard() {
  const [info, setInfo] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [metricsLoading, setMetricsLoading] = useState(false)

  useEffect(() => {
    axios.get('/api/model-info').then(r => setInfo(r.data))
  }, [])

  const loadMetrics = async () => {
    setMetricsLoading(true)
    try {
      const r = await axios.get('/api/metrics')
      setMetrics(r.data)
    } catch (e) {
      setMetrics({ error: e.response?.data?.error || e.message })
    }
    setMetricsLoading(false)
  }

  return (
    <div>
      <div className="page-header">
        <h2>📊 Model Dashboard</h2>
        <p>Model architecture, performance metrics, and training visualizations</p>
      </div>

      {/* Model Info */}
      {info && (
        <>
          <div className="metric-grid">
            <div className="metric-card">
              <div className="metric-value">{info.features}</div>
              <div className="metric-label">Features</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">{(info.threshold * 100).toFixed(1)}%</div>
              <div className="metric-label">Threshold</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">{info.variants.length}</div>
              <div className="metric-label">Dataset Variants</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">{info.training.num_clients}</div>
              <div className="metric-label">FL Clients</div>
            </div>
          </div>

          <div className="card-grid" style={{ marginBottom: 20 }}>
            {/* Architecture */}
            <div className="card">
              <div className="card-header"><h3>🏗️ Model Architecture</h3></div>
              <div style={{ fontFamily: 'monospace', fontSize: '0.85rem' }}>
                <p style={{ color: 'var(--info)', marginBottom: 8 }}>Input({info.architecture.input_dim})</p>
                {info.architecture.hidden_layers.map((units, i) => (
                  <p key={i} style={{ paddingLeft: 16 }}>
                    → Dense(<span style={{ color: 'var(--accent-hover)' }}>{units}</span>, {info.architecture.activation})
                    + Dropout({info.architecture.dropout_rate})
                    {info.architecture.l2_reg > 0 && ` + L2(${info.architecture.l2_reg})`}
                  </p>
                ))}
                <p style={{ paddingLeft: 16, color: 'var(--success)' }}>→ Dense(1, sigmoid) → P(fraud)</p>
              </div>
            </div>

            {/* Training Config */}
            <div className="card">
              <div className="card-header"><h3>⚙️ Training Configuration</h3></div>
              <table>
                <tbody>
                  {Object.entries(info.training).map(([k, v]) => (
                    <tr key={k}>
                      <td style={{ color: 'var(--text-muted)' }}>{k.replace(/_/g, ' ')}</td>
                      <td style={{ fontWeight: 600 }}>{String(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* Live Metrics */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header">
          <h3>📈 Live Performance Metrics</h3>
          <button className="btn btn-primary" onClick={loadMetrics} disabled={metricsLoading}>
            {metricsLoading ? <><span className="spinner"></span> Computing...</> : '🔄 Compute Metrics'}
          </button>
        </div>

        {metrics && !metrics.error && (
          <>
            <div className="metric-grid">
              <div className="metric-card">
                <div className="metric-value">{metrics.auc_roc}</div>
                <div className="metric-label">AUC-ROC</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">{(metrics.precision * 100).toFixed(1)}%</div>
                <div className="metric-label">Precision</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">{(metrics.recall * 100).toFixed(1)}%</div>
                <div className="metric-label">Recall</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">{metrics.separation}</div>
                <div className="metric-label">Separation</div>
              </div>
            </div>

            <div className="card-grid">
              <div className="card" style={{ background: 'var(--bg-primary)' }}>
                <h4 style={{ marginBottom: 12, fontSize: '0.9rem' }}>Confusion Matrix</h4>
                <table>
                  <thead><tr><th></th><th>Pred Legit</th><th>Pred Fraud</th></tr></thead>
                  <tbody>
                    <tr>
                      <td style={{ fontWeight: 600 }}>Actual Legit</td>
                      <td><span className="badge badge-success">TN: {metrics.confusion_matrix.true_negatives.toLocaleString()}</span></td>
                      <td><span className="badge badge-danger">FP: {metrics.confusion_matrix.false_positives.toLocaleString()}</span></td>
                    </tr>
                    <tr>
                      <td style={{ fontWeight: 600 }}>Actual Fraud</td>
                      <td><span className="badge badge-warning">FN: {metrics.confusion_matrix.false_negatives.toLocaleString()}</span></td>
                      <td><span className="badge badge-success">TP: {metrics.confusion_matrix.true_positives.toLocaleString()}</span></td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div className="card" style={{ background: 'var(--bg-primary)' }}>
                <h4 style={{ marginBottom: 12, fontSize: '0.9rem' }}>Prediction Distribution</h4>
                <div style={{ fontSize: '0.85rem' }}>
                  <p>Fraud mean: <strong style={{ color: 'var(--danger)' }}>{(metrics.fraud_mean_prediction * 100).toFixed(1)}%</strong></p>
                  <p>Legit mean: <strong style={{ color: 'var(--success)' }}>{(metrics.legit_mean_prediction * 100).toFixed(1)}%</strong></p>
                  <p>Test samples: <strong>{metrics.test_samples.toLocaleString()}</strong></p>
                </div>
              </div>
            </div>
          </>
        )}
        {metrics?.error && <p style={{ color: 'var(--danger)' }}>❌ {metrics.error}</p>}
      </div>

      {/* Result Images */}
      <div className="card-grid">
        {['roc_curve.png', 'confusion_matrix.png', 'training_history.png'].map(img => (
          <div className="card" key={img}>
            <div className="card-header"><h3>{img.replace('.png', '').replace(/_/g, ' ')}</h3></div>
            <div className="result-image">
              <img src={`/api/images/${img}`} alt={img} onError={e => { e.target.style.display = 'none' }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
