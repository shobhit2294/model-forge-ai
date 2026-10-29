import { StrictMode, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';
import './architecture.css';
import './prediction.css';

const initialForm = { goal: 'Build the best predictive model', target: '', metric: 'f1', minimum_score: '0.80', validation_strategy: 'holdout', max_retries: '2' };

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
  const [datasetProfile, setDatasetProfile] = useState(null);
  const [profiling, setProfiling] = useState(false);
  const [status, setStatus] = useState('');
  const [approving, setApproving] = useState(false);

  async function inspectDataset(selectedFile) {
    setFile(selectedFile);
    setResult(null);
    setDatasetProfile(null);
    setForm((current) => ({ ...current, target: '' }));
    if (!selectedFile) return;
    setProfiling(true);
    setStatus('Inspecting uploaded dataset...');
    try {
      const body = new FormData();
      body.append('file', selectedFile);
      const response = await fetch('/api/profile', { method: 'POST', body });
      const payload = await readResponse(response);
      if (!response.ok) throw new Error(payload.detail || 'Dataset profiling failed');
      setDatasetProfile(payload);
      setStatus(`Profiled ${payload.rows.toLocaleString()} rows and ${payload.columns} columns. Confirm the target below.`);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setProfiling(false);
    }
  }

  async function analyze(event) {
    event.preventDefault();
    if (!file) return;
    if (!datasetProfile || datasetProfile.filename !== file.name) {
      setStatus('Wait for dataset profiling to finish before starting training.');
      return;
    }
    if (!form.target) {
      setStatus('Confirm a target column, or explicitly choose profile-only analysis.');
      return;
    }
    setStatus('Launching AI Engineering Agent workflow...');
    const body = new FormData();
    Object.entries(form).forEach(([key, value]) => {
      body.append(key, key === 'target' && value === '__no_target__' ? '' : value);
    });
    body.append('target_confirmed', 'true');
    body.append('file', file);
    try {
      const response = await fetch('/api/analyze', { method: 'POST', body });
      const payload = await readResponse(response);
      if (!response.ok) throw new Error(payload.detail || 'Workflow failed');
      setResult(payload);
      setStatus('Agent workflow complete');
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function approveDeployment() {
    if (!result?.run_id) return;
    setApproving(true);
    setStatus('Waiting for deployment preparation...');
    try {
      const response = await fetch(`/api/runs/${result.run_id}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approved: true, reviewer: 'web-user' }),
      });
      const payload = await readResponse(response);
      if (!response.ok) throw new Error(payload.detail || 'Deployment approval failed');
      setResult(payload);
      setStatus(payload.deployment.status === 'approved' ? 'Deployment approved' : 'Deployment was not approved');
    } catch (error) {
      setStatus(error.message);
    } finally {
      setApproving(false);
    }
  }

  const update = (key) => (event) => setForm({ ...form, [key]: event.target.value });

  return (
    <main className="app-shell">
      <header>
        <div>
          <span className="eyebrow">AI ENGINEERING AGENT / ORCHESTRATOR</span>
          <h1>Intelligent ML<br /><em>from data to decision.</em></h1>
          <p className="intro">
            Give the AI Engineering Agent a dataset, a problem, and a measurable goal.
            Autonomous agents inspect, experiment, decide, package, and monitor — end to end.
          </p>
        </div>
        <div className="system-chip">
          <span className="pulse" />AGENT SYSTEM<br /><b>ONLINE</b>
        </div>
      </header>

      <AgentPipelineDiagram />

      <section className="work-grid">
        <form className="request-panel" onSubmit={analyze}>
          <div className="section-tag">01 / MISSION INPUT</div>
          <h2>Define the task</h2>

          <label>
            Dataset
            <input
              type="file"
              accept=".csv,.json,.jsonl,.ndjson,.xml,.xlsx,.xls,.parquet"
              onChange={(event) => inspectDataset(event.target.files[0])}
              required
            />
          </label>

          {datasetProfile && (
            <div className="decision-box">
              <small>DATASET PROFILE</small>
              <p>
                Detected {datasetProfile.format.toUpperCase()} data with {datasetProfile.rows.toLocaleString()} rows
                and {datasetProfile.columns} columns.
                {datasetProfile.candidate_target
                  ? ` Suggested target: ${datasetProfile.candidate_target}. Confirm the target before model training.`
                  : ' No likely target was found; choose one if labels are present, or choose profile-only analysis.'}
              </p>
            </div>
          )}

          <label>
            Engineering goal
            <input value={form.goal} onChange={update('goal')} placeholder="e.g. Predict customer churn" />
          </label>

          <label>
            Confirm prediction target
            <select value={form.target} onChange={update('target')} disabled={!datasetProfile || profiling}>
              <option value="">Choose a target or profile-only analysis</option>
              {datasetProfile?.column_profiles
                .filter((column) => column.data_type !== 'annotation_collection')
                .map((column) => <option key={column.name} value={column.name}>{column.name} ({column.data_type})</option>)}
              <option value="__no_target__">No target — profile only, no predictive model</option>
            </select>
          </label>
          <small className="status">
            {datasetProfile?.candidate_target
              ? `Suggested target: ${datasetProfile.candidate_target}. Select it to confirm, choose another column, or opt out of prediction.`
              : 'A supervised prediction model needs a labeled target column.'}
          </small>

          <div className="two-col">
            <label>
              Primary metric
              <select value={form.metric} onChange={update('metric')}>
                <option value="f1">F1 score</option>
                <option value="accuracy">Accuracy</option>
                <option value="precision">Precision</option>
                <option value="recall">Recall</option>
                <option value="r2">R²</option>
                <option value="rmse">RMSE</option>
                <option value="mae">MAE</option>
              </select>
            </label>
            <label>
              Minimum score
              <input type="number" step=".01" value={form.minimum_score} onChange={update('minimum_score')} />
            </label>
          </div>
          <small className="status">Validation scores estimate performance on held-out data; they do not guarantee perfect future predictions.</small>

          <label>
            Validation strategy
            <select value={form.validation_strategy} onChange={update('validation_strategy')}>
              <option value="holdout">Holdout (stratified for classification)</option>
              <option value="cross_validation">Cross-validation (up to 5 folds)</option>
            </select>
          </label>

          <label>
            Maximum stage retries <span style={{ fontWeight: 400, textTransform: 'none', opacity: 0.6 }}>(0–5)</span>
            <input type="number" min="0" max="5" step="1" value={form.max_retries} onChange={update('max_retries')} />
          </label>

          <button disabled={profiling || !datasetProfile || !form.target}>{profiling ? 'Profiling dataset...' : 'Run AI agents'} <span>→</span></button>
          <small className="status">{status || (file ? file.name : 'Upload a dataset to begin')}</small>
        </form>

        <section className="result-panel">
          {result
            ? <Result data={result} onApprove={approveDeployment} approving={approving} />
            : (
              <div className="empty">
                <span>02 / AGENT OUTPUT</span>
                <h2>Evidence before deployment.</h2>
                <p>
                  The agent system returns data quality checks, research methods,
                  experiment results, the quality-gate decision, and deployment
                  readiness — all in one workflow.
                </p>
                <ul className="agent-preview">
                  <li>🔍 <b>Data Agent</b> — profiling, cleaning, quality gate</li>
                  <li>📚 <b>Research Agent</b> — method selection from catalog</li>
                  <li>🤖 <b>Model Agent</b> — train &amp; compare candidates</li>
                  <li>⚖️ <b>Decision Agent</b> — score vs threshold gate</li>
                  <li>🚀 <b>DevOps Agent</b> — package &amp; deploy approved model</li>
                  <li>📡 <b>Monitoring Agent</b> — drift &amp; performance signals</li>
                </ul>
              </div>
            )
          }
        </section>
      </section>

      <footer>
        <span>FASTAPI · LANGGRAPH · SCIKIT-LEARN · PROMETHEUS</span>
        <span>AI ENGINEERING AGENT v0.1.0</span>
      </footer>
    </main>
  );
}

function AgentPipelineDiagram() {
  const agents = [
    { id: 'data',     label: 'DATA AGENT',     sub: 'Profile · Clean · Quality gate', color: 'data' },
    { id: 'research', label: 'RESEARCH AGENT',  sub: 'Methods · Baselines · Catalog',  color: 'research' },
    { id: 'model',    label: 'MODEL AGENT',     sub: 'Train · Compare · Evaluate',     color: 'model' },
    { id: 'decision', label: 'DECISION AGENT',  sub: 'Score gate · Best model',        color: 'decision' },
    { id: 'devops',   label: 'DEVOPS AGENT',    sub: 'Package · Artifact · Deploy',    color: 'devops' },
    { id: 'monitor',  label: 'MONITORING AGENT',sub: 'Drift · Latency · Performance',  color: 'monitor' },
  ];

  return (
    <section className="architecture" aria-label="AI Engineering Agent pipeline">
      <div className="architecture-heading">
        <span>AI ENGINEERING AGENT / SYSTEM MAP</span>
        <b>Dataset → Agents → Production model → Continuous improvement</b>
      </div>
      <div className="diagram-canvas">
        <div className="diagram-node client">
          <strong>USER / CLIENT</strong>
          <span>Upload dataset · Set goal · Trigger agents</span>
        </div>
        <div className="connector vertical" />
        <div className="diagram-node orchestrator">
          <strong>AI ENGINEERING AGENT</strong>
          <span>LangGraph Orchestrator · FastAPI · Python</span>
        </div>
        <div className="connector vertical" />
        <div className="parallel-agents">
          {agents.slice(0, 3).map((a) => (
            <div key={a.id} className={`diagram-node agent ${a.color}`}>
              <strong>{a.label}</strong>
              <span>{a.sub}</span>
            </div>
          ))}
        </div>
        <div className="connector vertical" />
        <div className="diagram-node decision">
          <strong>DECISION AGENT</strong>
          <span>Compare results · Quality gate · Next action</span>
        </div>
        <div className="decision-branch">
          <div className="branch-line" />
          <div className="diagram-node branch blocked">
            <strong>BLOCKED</strong>
            <span>Re-experiment ↺</span>
          </div>
          <div className="diagram-node branch approved">
            <strong>APPROVED</strong>
            <span>DevOps Agent →</span>
          </div>
        </div>
        <div className="connector vertical" />
        <div className="diagram-node production">
          <strong>PRODUCTION MODEL</strong>
          <span>Joblib artifact · FastAPI /predict</span>
        </div>
        <div className="connector vertical" />
        <div className="diagram-node monitoring">
          <strong>MONITORING AGENT</strong>
          <span>Drift · Errors · Latency · Performance</span>
        </div>
        <div className="feedback">
          <span>Issue detected?</span>
          <b>↺ DECISION AGENT · RETRAIN / ROLLBACK</b>
        </div>
      </div>
    </section>
  );
}

function Result({ data, onApprove, approving }) {
  const evaluation = data.evaluation;
  const featureColumns = data.profile.column_profiles.filter(
    (col) => col.name !== data.profile.target && col.data_type !== 'annotation_collection'
  );

  return (
    <>
      <div className="result-title">
        <span>RUN {data.run_id}</span>
        <strong className={data.decision.status}>{data.decision.status.replace('_', ' ').toUpperCase()}</strong>
      </div>

      <div className="summary">
        <div><small>TASK TYPE</small><b>{data.profile.task}</b></div>
        <div><small>ROWS</small><b>{data.profile.rows.toLocaleString()}</b></div>
        <div><small>BEST MODEL</small><b>{data.decision.best_model || '—'}</b></div>
      </div>

      {data.decision.best_score != null && (
        <div className="summary" style={{ borderTop: 'none', paddingTop: 0 }}>
          <div><small>SCORE</small><b>{data.decision.best_score}</b></div>
          <div><small>METRIC</small><b>{data.decision.metric?.toUpperCase()}</b></div>
          <div><small>THRESHOLD</small><b>{data.decision.threshold ?? '—'}</b></div>
        </div>
      )}
      {evaluation?.status === 'complete' && (
        <div className="decision-box">
          <small>VALIDATION IS AN ESTIMATE</small>
          <p>
            Scores are measured using {evaluation.validation?.strategy || 'the selected validation strategy'}.
            They describe performance on validation data, not a guarantee of perfect predictions on future data.
          </p>
        </div>
      )}

      <div className="decision-box">
        <small>DECISION AGENT</small>
        <p>{data.decision.next_step}</p>
      </div>

      {data.deployment?.status === 'awaiting_approval' && (
        <div className="decision-box">
          <small>HUMAN APPROVAL REQUIRED</small>
          <p>Review the selected model, validation score, and quality checks before creating a deployable artifact.</p>
          <button type="button" onClick={onApprove} disabled={approving}>{approving ? 'Preparing deployment...' : 'Approve deployment →'}</button>
        </div>
      )}

      <h2>Agent pipeline</h2>
      <div className="agent-list">
        {data.stages.map((stage) => (
          <div className="agent-row" key={stage.agent}>
            <span>{stage.agent}</span>
            <b className={`status-badge status-${stage.status}`}>{stage.status}</b>
          </div>
        ))}
      </div>

      {evaluation?.status === 'complete' && (
        <>
          <h2>Experiments</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>MODEL</th>
                  {data.profile.task === 'regression'
                    ? <><th>MAE</th><th>RMSE</th><th>R²</th></>
                    : <><th>F1</th><th>PRECISION</th><th>RECALL</th><th>ACCURACY</th></>
                  }
                  <th>WINNER</th>
                </tr>
              </thead>
              <tbody>
                {evaluation.models.map((m) => (
                  <tr className={m.model === evaluation.best_model ? 'winner' : ''} key={m.model}>
                    <td>{m.model}</td>
                    {data.profile.task === 'regression'
                      ? <><td>{m.mae?.toFixed(4) ?? '—'}</td><td>{m.rmse?.toFixed(4) ?? '—'}</td><td>{m.r2?.toFixed(4) ?? '—'}</td></>
                      : <><td>{m.f1?.toFixed(4) ?? '—'}</td><td>{m.precision?.toFixed(4) ?? '—'}</td><td>{m.recall?.toFixed(4) ?? '—'}</td><td>{m.accuracy?.toFixed(4) ?? '—'}</td></>
                    }
                    <td>{m.model === evaluation.best_model ? '✓' : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {data.deployment?.status === 'approved' && (
        <PredictionPanel runId={data.run_id} columns={featureColumns} />
      )}

      {data.quality?.checks && (
        <div className="quality">
          <small>DATA QUALITY CHECKS</small>
          {data.quality.checks.map((check) => (
            <p key={check}>✓ {check}</p>
          ))}
          {data.quality.duplicates > 0 && (
            <p style={{ color: '#c84b3a' }}>⚠ {data.quality.duplicates} duplicate row(s) detected</p>
          )}
        </div>
      )}

      {data.profile.warnings?.length > 0 && (
        <div className="quality" style={{ borderColor: '#e1b65a', background: '#fdf6e3' }}>
          <small>DATASET WARNINGS</small>
          {data.profile.warnings.map((w) => <p key={w}>⚠ {w}</p>)}
        </div>
      )}

      {data.orchestration && (
        <div className="quality" style={{ borderColor: '#8bc0e6', background: '#dbeaf8' }}>
          <small>LANGGRAPH ORCHESTRATION</small>
          <p>Engine: {data.orchestration.engine} · Checkpointed: <b>{data.orchestration.checkpointed ? 'yes' : 'no'}</b> · Status: <b>{data.orchestration.next_action}</b> · Max retries per stage: {data.orchestration.retry_limit}</p>
        </div>
      )}

      {data.execution_log?.length > 0 && (
        <div className="quality">
          <small>WORKFLOW EXECUTION LOG</small>
          {data.execution_log.map((entry, index) => (
            <p key={`${entry.stage}-${entry.attempt}-${index}`}>
              {entry.stage} · attempt {entry.attempt} · {entry.status}
              {entry.error ? ` · ${entry.error}` : ''}
              {entry.duration_ms != null ? ` · ${entry.duration_ms} ms` : ''}
            </p>
          ))}
        </div>
      )}

      {data.errors?.length > 0 && (
        <div className="quality">
          <small>WORKFLOW ERRORS</small>
          {data.errors.map((error, index) => <p key={`${error.stage}-${index}`}>{error.stage}: {error.error}</p>)}
        </div>
      )}
    </>
  );
}

function PredictionPanel({ runId, columns }) {
  const [values, setValues] = useState({});
  const [output, setOutput] = useState(null);
  const [busy, setBusy] = useState(false);

  async function predict(event) {
    event.preventDefault();
    setBusy(true);
    try {
      const response = await fetch(`/api/runs/${runId}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values),
      });
      const payload = await readResponse(response);
      setOutput(response.ok ? payload : { error: payload.detail || 'Prediction failed' });
    } catch (error) {
      setOutput({ error: error.message });
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="prediction-panel" onSubmit={predict}>
      <div className="prediction-heading">
        <small>DEVOPS AGENT / LIVE INFERENCE</small>
        <strong>MODEL DEPLOYED</strong>
      </div>
      <p>Enter feature values below. The AI agent calls the trained production pipeline and returns a prediction instantly.</p>
      <div className="prediction-grid">
        {columns.map((col) => (
          <label key={col.name}>
            {col.name}
            <input
              required={col.missing === 0}
              type={col.data_type === 'numeric' ? 'number' : 'text'}
              placeholder={col.sample?.[0] || 'optional'}
              value={values[col.name] || ''}
              onChange={(e) => setValues({ ...values, [col.name]: e.target.value })}
            />
          </label>
        ))}
      </div>
      <button disabled={busy}>{busy ? 'Agent predicting...' : 'Run prediction →'}</button>
      {output && (
        <div className="prediction-output">
          <small>PREDICTION RESULT</small>
          <b>{output.error || output.prediction}</b>
          {output.probability !== undefined && <span>confidence {(output.probability * 100).toFixed(1)}%</span>}
        </div>
      )}
    </form>
  );
}

createRoot(document.getElementById('root')).render(<StrictMode><App /></StrictMode>);