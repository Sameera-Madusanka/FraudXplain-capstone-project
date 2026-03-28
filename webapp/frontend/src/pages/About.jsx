export default function About() {
  const challenges = [
    { id: 1, name: 'SMOTE + FL', cause: 'Synthetic data inconsistency across clients', fix: 'Removed SMOTE entirely', impact: '0.61 → 0.78' },
    { id: 2, name: 'Class Weight Instability', cause: 'Too few fraud per client with IID split', fix: 'Balanced distribution (50:50)', impact: 'Stabilized' },
    { id: 3, name: 'Focal Loss + FL', cause: 'Incompatible gradient scales during aggregation', fix: 'Reverted to BCE', impact: 'Stabilized' },
    { id: 4, name: 'BatchNorm + FL', cause: 'Local batch statistics averaged incorrectly', fix: 'Removed BatchNorm, use Dense+Dropout', impact: 'Stabilized' },
    { id: 5, name: 'IID Starves Fraud', cause: '1% fraud per client — insufficient', fix: 'Balanced 50:50 per client', impact: '→ 0.70' },
    { id: 6, name: 'DP Noise Destroys Weights ⭐', cause: 'Gaussian noise × 100 injections', fix: 'Disabled DP', impact: '0.64 → 0.84' },
    { id: 7, name: 'Hardcoded 0.5 Threshold', cause: 'Wrong decision boundary', fix: 'F1-optimized threshold from file', impact: 'FP/FN fixed' },
    { id: 8, name: 'SMOTE on Test Data', cause: 'Synthetic test ≠ real distribution', fix: 'balance_classes=False for test', impact: 'Accurate eval' },
    { id: 9, name: 'FP > TP Mathematically', cause: '1% fraud rate → precision < 50%', fix: 'Multi-variant + deeper model', impact: 'Improving' },
    { id: 10, name: 'Extra Columns in Variants', cause: 'Variant III/V have x1, x2', fix: 'Auto-drop extra columns', impact: 'No error' },
    { id: 11, name: 'Multi-variant Unbalanced', cause: 'No per-client balancing in new path', fix: 'Added subsampling in load_multi_variant', impact: '0.83 → 0.88' },
  ]

  const references = [
    'McMahan et al. (2017) — "Communication-Efficient Learning from Decentralized Data" (FedAvg)',
    'Jesus et al. (2022) — "Turning the Tables: Biased, Imbalanced, Dynamic Datasets" (NeurIPS)',
    'Abadi et al. (2016) — "Deep Learning with Differential Privacy" (CCS)',
    'Li et al. (2022) — "FedBN: FL on Non-IID Features via Local Batch Normalization" (ICLR)',
    'Chawla et al. (2002) — "SMOTE: Synthetic Minority Over-sampling Technique" (JAIR)',
    'Lin et al. (2017) — "Focal Loss for Dense Object Detection" (ICCV)',
    'Lipton et al. (2014) — "Optimal Thresholding to Maximize F1 Measure" (ECML PKDD)',
    'Zhao et al. (2018) — "Federated Learning with Non-IID Data"',
    'Wei et al. (2020) — "Federated Learning with Differential Privacy" (IEEE TIFS)',
  ]

  return (
    <div>
      <div className="page-header">
        <h2>ℹ️ About FraudXplain</h2>
        <p>Project overview, challenges encountered, and research references</p>
      </div>

      {/* About The Research Project */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header"><h3>About This Project</h3></div>
        <div style={{ fontSize: '0.95rem', lineHeight: 1.8 }}>
          <p>
            This web application presents the implementation of the final year research project titled <strong>“Developing a Privacy-Preserving Explainability Framework for Generating Actionable Counterfactual Explanations in Federated Financial Fraud Detection.”</strong>
          </p>
          <p style={{ marginTop: 12 }}>
            The project focuses on addressing critical challenges in financial AI, particularly data privacy, model transparency, and trust. It demonstrates how Federated Learning can be combined with Explainable AI techniques to enable collaborative fraud detection without sharing sensitive data, while also providing meaningful and privacy-preserving explanations for model decisions.
          </p>
          <p style={{ marginTop: 12 }}>
            Developed by <strong>Sameera Madusanka (w1871882)</strong> as part of the BSc (Hons) in Computer Science degree at the Informatics Institute of Technology (IIT), affiliated with the University of Westminster, UK, this application serves as a prototype to showcase the system’s core functionalities, experimental outcomes, and practical relevance in real-world financial environments.
          </p>
        </div>
      </div>

      {/* Overview */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header"><h3>Project Overview</h3></div>
        <div style={{ fontSize: '0.9rem', lineHeight: 1.8 }}>
          <p><strong>FraudXplain</strong> is a privacy-preserving fraud detection system that combines:</p>
          <ul style={{ paddingLeft: 20, marginTop: 8 }}>
            <li><strong>Federated Learning</strong> — distributed training across 6 institutions without sharing raw data</li>
            <li><strong>Constrained Counterfactual Explanations</strong> — actionable explanations that protect sensitive attributes</li>
            <li><strong>Multi-Variant Training</strong> — 6 heterogeneous datasets for realistic data distribution</li>
          </ul>
          <p style={{ marginTop: 12 }}>
            <strong>Novel Contribution:</strong> First system to combine FL with constrained CFs, providing formal privacy guarantees in both training and explanations.
          </p>
        </div>
      </div>

      {/* Challenges */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-header"><h3>🔧 Challenges & Solutions ({challenges.length})</h3></div>
        <div className="table-container">
          <table>
            <thead>
              <tr><th>#</th><th>Challenge</th><th>Root Cause</th><th>Solution</th><th>AUC Impact</th></tr>
            </thead>
            <tbody>
              {challenges.map(c => (
                <tr key={c.id}>
                  <td>{c.id}</td>
                  <td style={{ fontWeight: 600 }}>{c.name}</td>
                  <td style={{ color: 'var(--text-secondary)' }}>{c.cause}</td>
                  <td style={{ color: 'var(--success)' }}>{c.fix}</td>
                  <td><span className="badge badge-info">{c.impact}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* References */}
      <div className="card">
        <div className="card-header"><h3>📚 Key References</h3></div>
        <ul style={{ listStyle: 'none', fontSize: '0.85rem' }}>
          {references.map((r, i) => (
            <li key={i} style={{ padding: '6px 0', borderBottom: '1px solid var(--border)' }}>
              <span style={{ color: 'var(--accent-hover)', marginRight: 8 }}>[{i + 1}]</span>
              {r}
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
