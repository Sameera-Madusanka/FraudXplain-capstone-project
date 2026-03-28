export default function SystemArchitecture() {
  return (
    <div>
      <div className="page-header">
        <h2>🏗️ System Architecture</h2>
        <p>How FraudXplain combines Federated Learning with Privacy-Preserving Explainability</p>
      </div>

      {/* FL Pipeline */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header"><h3>Federated Learning Pipeline</h3></div>
        <div className="arch-flow">
          {['Client 1\nBase.csv', 'Client 2\nVariant I', 'Client 3\nVariant II', 'Client 4\nVariant III', 'Client 5\nVariant IV', 'Client 6\nVariant V'].map((c, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div className="arch-block">
                <div className="block-title">🏦</div>
                <div className="block-detail" style={{ whiteSpace: 'pre-line' }}>{c}</div>
              </div>
              {i < 5 && <span style={{ color: 'var(--text-muted)' }}>|</span>}
            </div>
          ))}
        </div>
        <div style={{ display: 'flex', justifyContent: 'center', margin: '8px 0' }}>
          <span className="arch-arrow">↓ Send Local Weights ↓</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'center', margin: '12px 0' }}>
          <div className="arch-block highlight">
            <div className="block-title">🖥️ FedAvg Server</div>
            <div className="block-detail">Aggregate → Global Model</div>
          </div>
        </div>
        <div style={{ display: 'flex', justifyContent: 'center', margin: '8px 0' }}>
          <span className="arch-arrow">↓ Broadcast Global Weights ↓</span>
        </div>
        <p style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.82rem' }}>
          Repeat for 30 rounds · Raw data never leaves clients
        </p>
      </div>

      {/* Model Architecture */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header"><h3>Neural Network Architecture</h3></div>
        <div className="arch-flow">
          {[
            { title: 'Input', detail: '31 features' },
            { title: 'Dense(256)', detail: 'ReLU + L2 + Drop' },
            { title: 'Dense(128)', detail: 'ReLU + L2 + Drop' },
            { title: 'Dense(64)', detail: 'ReLU + L2 + Drop' },
            { title: 'Dense(32)', detail: 'ReLU + L2 + Drop' },
            { title: 'Dense(1)', detail: 'Sigmoid → P(fraud)' },
          ].map((b, i, arr) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div className={`arch-block ${i === 0 || i === arr.length - 1 ? 'highlight' : ''}`}>
                <div className="block-title">{b.title}</div>
                <div className="block-detail">{b.detail}</div>
              </div>
              {i < arr.length - 1 && <span className="arch-arrow">→</span>}
            </div>
          ))}
        </div>
      </div>

      {/* Privacy Guarantees */}
      <div className="card-grid" style={{ marginBottom: 20 }}>
        <div className="card">
          <div className="card-header"><h3>🔒 Training Privacy</h3></div>
          <ul style={{ listStyle: 'none', fontSize: '0.88rem' }}>
            <li style={{ padding: '6px 0' }}>✅ Raw data never leaves client institutions</li>
            <li style={{ padding: '6px 0' }}>✅ Only model weights are shared</li>
            <li style={{ padding: '6px 0' }}>✅ FedAvg aggregation (weighted by sample count)</li>
            <li style={{ padding: '6px 0' }}>✅ 6 independent institutions → natural non-IID</li>
          </ul>
        </div>
        <div className="card">
          <div className="card-header"><h3>🔐 Explanation Privacy</h3></div>
          <div style={{ marginBottom: 12 }}>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: 8 }}>Protected attributes (NEVER changed):</p>
            <div className="protected-list">
              {['income', 'customer_age', 'employment_status', 'housing_status', 'date_of_birth_distinct_emails_4w', 'foreign_request'].map(a => (
                <span className="protected-tag" key={a}>🔒 {a}</span>
              ))}
            </div>
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
            Counterfactual explanations only suggest changes to actionable attributes.
          </p>
        </div>
      </div>

      {/* Balanced Distribution */}
      <div className="card">
        <div className="card-header"><h3>⚖️ Balanced Distribution Strategy</h3></div>
        <div className="card-grid">
          <div>
            <h4 style={{ color: 'var(--danger)', fontSize: '0.88rem', marginBottom: 8 }}>❌ Without Balancing (IID)</h4>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Each client gets ~1% fraud → model can't learn fraud patterns → AUC stuck at 0.64
            </p>
          </div>
          <div>
            <h4 style={{ color: 'var(--success)', fontSize: '0.88rem', marginBottom: 8 }}>✅ With Balancing (50:50)</h4>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Each client gets 50% fraud + 50% legit → model learns patterns → AUC = 0.88
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
