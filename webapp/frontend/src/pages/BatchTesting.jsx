import { useState } from 'react'
import axios from 'axios'

export default function BatchTesting() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [fileName, setFileName] = useState('')

  const handleUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    setFileName(file.name)
    setLoading(true)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const r = await axios.post('/api/batch-predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setResults(r.data)
    } catch (e) {
      setResults({ error: e.response?.data?.error || e.message })
    }
    setLoading(false)
  }

  const downloadCSV = () => {
    if (!results?.predictions) return
    const rows = ['Row,Fraud Probability,Is Fraud,Risk Level']
    results.predictions.forEach(p => {
      rows.push(`${p.row},${p.fraud_probability},${p.is_fraud},${p.risk_level}`)
    })
    const blob = new Blob([rows.join('\n')], { type: 'text/csv' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'fraud_predictions.csv'
    a.click()
  }

  return (
    <div>
      <div className="page-header">
        <h2>📋 Batch Testing</h2>
        <p>Upload a CSV file with transaction data for batch fraud predictions</p>
      </div>

      {/* Upload Zone */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header"><h3>Upload CSV</h3></div>
        <label className="upload-zone">
          <div className="upload-icon">📁</div>
          <p style={{ fontWeight: 600, marginBottom: 4 }}>
            {fileName || 'Click to upload CSV file'}
          </p>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
            CSV must have at least 31 numeric columns (feature values)
          </p>
          <input type="file" accept=".csv" onChange={handleUpload} />
        </label>
      </div>

      {loading && (
        <div className="card">
          <div className="loading"><span className="spinner"></span> Processing batch predictions...</div>
        </div>
      )}

      {results?.error && (
        <div className="card" style={{ borderColor: 'var(--danger)' }}>
          <p style={{ color: 'var(--danger)' }}>❌ {results.error}</p>
        </div>
      )}

      {results && !results.error && (
        <>
          {/* Summary */}
          <div className="metric-grid">
            <div className="metric-card">
              <div className="metric-value">{results.total}</div>
              <div className="metric-label">Total Transactions</div>
            </div>
            <div className="metric-card">
              <div className="metric-value" style={{ background: 'var(--gradient-danger)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                {results.fraud_detected}
              </div>
              <div className="metric-label">Fraud Detected</div>
            </div>
            <div className="metric-card">
              <div className="metric-value" style={{ background: 'var(--gradient-success)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                {results.legitimate}
              </div>
              <div className="metric-label">Legitimate</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">{(results.threshold * 100).toFixed(1)}%</div>
              <div className="metric-label">Threshold</div>
            </div>
          </div>

          {/* Download */}
          <div style={{ marginBottom: 16 }}>
            <button className="btn btn-primary" onClick={downloadCSV}>📥 Download Results CSV</button>
          </div>

          {/* Results Table */}
          <div className="card">
            <div className="card-header"><h3>Predictions</h3></div>
            <div className="table-container">
              <table>
                <thead>
                  <tr><th>Row</th><th>Fraud Probability</th><th>Classification</th><th>Risk</th></tr>
                </thead>
                <tbody>
                  {results.predictions.slice(0, 100).map(p => (
                    <tr key={p.row}>
                      <td>{p.row}</td>
                      <td>{(p.fraud_probability * 100).toFixed(2)}%</td>
                      <td>
                        <span className={`badge ${p.is_fraud ? 'badge-danger' : 'badge-success'}`}>
                          {p.is_fraud ? '🚨 Fraud' : '✅ Legit'}
                        </span>
                      </td>
                      <td>
                        <span className={`badge ${p.risk_level === 'High' ? 'badge-danger' : p.risk_level === 'Medium' ? 'badge-warning' : 'badge-success'}`}>
                          {p.risk_level}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {results.predictions.length > 100 && (
                <p style={{ padding: 12, color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                  Showing first 100 of {results.predictions.length} rows. Download CSV for full results.
                </p>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
