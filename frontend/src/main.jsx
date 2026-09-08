import { StrictMode, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';
import './architecture.css';
import './prediction.css';

const initialForm = { goal: 'Predict customer churn', target: '', metric: 'f1', minimum_score: '0.80' };

async function readResponse(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text.slice(0, 200) };
  }
}

function App() {
  const [form, setForm] = useState(initialForm);
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState('');

  async function analyze(event) {
    event.preventDefault();
    if (!file) return;
    setStatus('Running engineering workflow...');
    const body = new FormData();
    Object.entries(form).forEach(([key, value]) => body.append(key, value));
    body.append('file', file);
    try {
      const response = await fetch('/api/analyze', { method: 'POST', body });
      const payload = await readResponse(response);
      if (!response.ok) throw new Error(payload.detail || 'Workflow failed');
      setResult(payload);
      setStatus('Workflow complete');
    } catch (error) {
      setStatus(error.message);
    }
  }

  const update = (key) => (event) => setForm({ ...form, [key]: event.target.value });
  return <main className="app-shell">
    <header><div><span className="eyebrow">AI ENGINEERING AGENT / ORCHESTRATOR</span><h1>From raw data<br /><em>to a production decision.</em></h1><p className="intro">Give the system a problem, a dataset, and a measurable requirement. The agents inspect, experiment, decide, package, and monitor.</p></div><div className="system-chip"><span className="pulse" />LOCAL SYSTEM<br /><b>READY</b></div></header>
    <ArchitectureDiagram />
    <section className="work-grid">
      <form className="request-panel" onSubmit={analyze}><div className="section-tag">01 / USER REQUEST</div><h2>Define the problem</h2><label>Dataset<input type="file" accept=".csv,.json,.jsonl,.ndjson,.xlsx,.xls,.parquet" onChange={(event) => setFile(event.target.files[0])} required /></label><label>Business goal<input value={form.goal} onChange={update('goal')} /></label><label>Target column<input value={form.target} onChange={update('target')} placeholder="churn, price, label" /></label><div className="two-col"><label>Primary metric<select value={form.metric} onChange={update('metric')}><option value="f1">F1 score</option><option value="accuracy">Accuracy</option><option value="precision">Precision</option><option value="recall">Recall</option><option value="rmse">RMSE</option></select></label><label>Minimum score<input type="number" min="0" max="1" step=".01" value={form.minimum_score} onChange={update('minimum_score')} /></label></div><button>Run full workflow <span>→</span></button><small className="status">{status || (file ? file.name : 'Select a dataset to begin')}</small></form>
      <section className="result-panel">{result ? <Result data={result} /> : <div className="empty"><span>02 / SYSTEM OUTPUT</span><h2>Evidence before deployment.</h2><p>The orchestrator will return data quality, research methods, experiment results, the quality-gate decision, and deployment state here.</p></div>}</section>
    </section>
    <footer><span>FASTAPI · LANGGRAPH · SCIKIT-LEARN · DOCKER</span><span>PROMETHEUS /metrics</span></footer>
  </main>;
}

function ArchitectureDiagram() {
  return <section className="architecture" aria-label="AI engineering agent architecture"><div className="architecture-heading"><span>ORCHESTRATOR / SYSTEM MAP</span><b>Requirement → model → continuous improvement</b></div><div className="diagram-canvas"><div className="diagram-node client"><strong>USER / CLIENT</strong><span>Web / API / CLI</span></div><div className="connector vertical" /><div className="diagram-node orchestrator"><strong>AI ENGINEERING AGENT</strong><span>ORCHESTRATOR</span></div><div className="connector vertical" /><div className="parallel-agents"><div className="diagram-node agent data"><strong>DATA AGENT</strong><span>Clean · EDA · Features</span></div><div className="diagram-node agent model"><strong>MODEL AGENT</strong><span>Train · Tune · Evaluate</span></div><div className="diagram-node agent research"><strong>RESEARCH AGENT</strong><span>Papers · Methods · Baselines</span></div></div><div className="connector vertical" /><div className="diagram-node decision"><strong>DECISION AGENT</strong><span>Compare · Next step · Quality gate</span></div><div className="decision-branch"><div className="branch-line" /><div className="diagram-node branch blocked"><strong>NOT GOOD</strong><span>New experiment ↺</span></div><div className="diagram-node branch approved"><strong>APPROVED</strong><span>Deployment Agent</span></div></div><div className="connector vertical" /><div className="diagram-node production"><strong>PRODUCTION MODEL</strong><span>API + Container</span></div><div className="connector vertical" /><div className="diagram-node monitoring"><strong>MONITORING</strong><span>Drift · Latency · Errors · Performance</span></div><div className="feedback"><span>Problem detected?</span><b>↺ DECISION AGENT · RETRAIN / ROLLBACK</b></div></div></section>;
}

function Result({ data }) {
  const evaluation = data.evaluation;
  const featureColumns = data.profile.column_profiles.filter((column) => column.name !== data.profile.target && column.data_type !== 'annotation_collection');
  return <><div className="result-title"><span>RUN {data.run_id}</span><strong className={data.decision.status}>{data.decision.status.replace('_', ' ')}</strong></div><div className="summary"><div><small>TASK</small><b>{data.profile.task}</b></div><div><small>ROWS</small><b>{data.profile.rows.toLocaleString()}</b></div><div><small>DECISION</small><b>{data.decision.best_model || 'Needs data'}</b></div></div><div className="decision-box"><small>DECISION AGENT</small><p>{data.decision.next_step}</p></div><h2>Agent evidence</h2><div className="agent-list">{data.stages.map((stage) => <div className="agent-row" key={stage.agent}><span>{stage.agent}</span><b>{stage.status}</b></div>)}</div>{evaluation?.status === 'complete' && <><h2>Experiments</h2><div className="table-wrap"><table><thead><tr><th>MODEL</th><th>F1</th><th>PRECISION</th><th>RECALL</th></tr></thead><tbody>{evaluation.models.map((model) => <tr className={model.model === evaluation.best_model ? 'winner' : ''} key={model.model}><td>{model.model}</td><td>{model.f1?.toFixed(4) || '-'}</td><td>{model.precision?.toFixed(4) || '-'}</td><td>{model.recall?.toFixed(4) || '-'}</td></tr>)}</tbody></table></div></>}{data.deployment?.status === 'approved' && <PredictionPanel runId={data.run_id} columns={featureColumns} />}{data.quality?.checks && <div className="quality"><small>DATA QUALITY</small>{data.quality.checks.map((check) => <p key={check}>✓ {check}</p>)}</div>}</>;
}

function PredictionPanel({ runId, columns }) {
  const [values, setValues] = useState({});
  const [output, setOutput] = useState(null);
  const [busy, setBusy] = useState(false);
  async function predict(event) {
    event.preventDefault();
    setBusy(true);
    try {
      const response = await fetch(`/api/runs/${runId}/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(values) });
      const payload = await readResponse(response);
      setOutput(response.ok ? payload : { error: payload.detail || 'Prediction failed' });
    } catch (error) {
      setOutput({ error: error.message });
    } finally {
      setBusy(false);
    }
  }
  return <form className="prediction-panel" onSubmit={predict}><div className="prediction-heading"><small>DEPLOYED MODEL / LIVE INFERENCE</small><strong>READY</strong></div><p>Enter feature values. This calls the trained production pipeline automatically.</p><div className="prediction-grid">{columns.map((column) => <label key={column.name}>{column.name}<input required={column.missing === 0} type={column.data_type === 'numeric' ? 'number' : 'text'} placeholder={column.sample?.[0] || 'optional'} value={values[column.name] || ''} onChange={(event) => setValues({ ...values, [column.name]: event.target.value })} /></label>)}</div><button disabled={busy}>{busy ? 'Predicting...' : 'Run prediction →'}</button>{output && <div className="prediction-output"><small>PREDICTION</small><b>{output.error || output.prediction}</b>{output.probability !== undefined && <span>probability {output.probability}</span>}</div>}</form>;
}

createRoot(document.getElementById('root')).render(<StrictMode><App /></StrictMode>);