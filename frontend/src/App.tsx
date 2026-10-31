import { useEffect, useState, useRef } from 'react';
import {
  Activity,
  Shield,
  Network,
  FlaskConical,
  Database,
  AlertTriangle,
  RefreshCw,
  Upload,
  RadioTower,
  ChevronRight,
  CheckCircle2,
  Terminal,
  BrainCircuit,
  Bot,
  Settings,
  Sparkles,
  ExternalLink,
  ShieldAlert,
  FileSpreadsheet,
  FolderOpen,
  LogOut,
  User as UserIcon,
  Lock,
  Server,
  AlertOctagon,
  Check,
} from 'lucide-react';
import {
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  Legend,
  CartesianGrid,
} from 'recharts';
import { User } from 'firebase/auth';
import { collection, addDoc, doc, setDoc } from 'firebase/firestore';
import { db, testConnection, handleFirestoreError, OperationType } from './services/firebase';
import {
  initAuth,
  googleSignIn,
  logout,
  openGooglePicker,
  downloadDriveFile,
  exportToGoogleSheets,
} from './services/workspace';

interface Client {
  id: number;
  name: string;
  status: 'HEALTHY' | 'QUARANTINED' | string;
  trust: number;
  samples: number;
  l2_norm?: number;
  mean_cosine?: number;
  deviation?: number;
  robust_score?: number;
  is_flagged?: boolean;
}

interface TestMetrics {
  loss?: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  fpr?: number;
  confusion?: { tp: number; tn: number; fp: number; fn: number };
}

interface LossDataPoint {
  round: number;
  train_loss: number;
  val_loss: number;
  test_loss: number;
}

interface ExperimentResult {
  round: number;
  partition?: 'iid' | 'non_iid' | string;
  attack: string;
  malicious_client: number | null;
  aggregation: string;
  flagged_clients: number[];
  detected_attacker: boolean | null;
  accepted_clients: number[];
  baseline?: TestMetrics;
  poisoned_fedavg?: TestMetrics;
  robust_aggregation?: TestMetrics;
  validation: TestMetrics;
  test: TestMetrics;
  loss_history?: LossDataPoint[];
  client_evidence: Client[];
  note: string;
}

interface DatasetSummary {
  name: string;
  synthetic: boolean;
  train_samples: number;
  val_samples: number;
  test_samples: number;
  features: number;
  class_counts: Record<string, number>;
}

interface EventLog {
  time: string;
  event: string;
  details: Record<string, unknown>;
  operator?: string;
}

interface State {
  dataset: DatasetSummary | null;
  clients: Client[];
  events: EventLog[];
  round: number;
  last_experiment: ExperimentResult | null;
  loss_history?: LossDataPoint[];
  persistence?: string;
  database?: {
    mode: string;
    project_id: string;
    database_id: string;
    is_firestore: boolean;
  };
  operator?: {
    email: string;
    role: string;
    isAuthenticated: boolean;
  };
}

export const DEFAULT_LOCAL_BACKEND = 'http://localhost:8000';

export function validateApiEndpoint(url: string): { valid: boolean; error?: string; cleanUrl: string } {
  const trimmed = url.trim();
  // An empty string is accepted to use the same-origin relative /api proxy
  if (trimmed === '') {
    return { valid: true, cleanUrl: '' };
  }

  // Reject invalid numeric strings like "123"
  if (/^\d+$/.test(trimmed)) {
    return {
      valid: false,
      error: `Invalid numeric value '${trimmed}'. An API endpoint must be a valid HTTP or HTTPS URL (e.g. http://localhost:8000 or https://your-service.run.app).`,
      cleanUrl: DEFAULT_LOCAL_BACKEND,
    };
  }

  // Enforce protocol
  if (!trimmed.startsWith('http://') && !trimmed.startsWith('https://')) {
    return {
      valid: false,
      error: 'Protocol missing. The URL must start with http:// or https:// (e.g. http://localhost:8000 or https://your-cloud-run-url.a.run.app).',
      cleanUrl: DEFAULT_LOCAL_BACKEND,
    };
  }

  try {
    const parsed = new URL(trimmed);
    if (!parsed.hostname) {
      return {
        valid: false,
        error: 'Malformed URL: hostname is missing.',
        cleanUrl: DEFAULT_LOCAL_BACKEND,
      };
    }
    const clean = parsed.origin + (parsed.pathname === '/' ? '' : parsed.pathname.replace(/\/$/, ''));
    return { valid: true, cleanUrl: clean };
  } catch (err: any) {
    return {
      valid: false,
      error: `Malformed URL format: ${err.message}`,
      cleanUrl: DEFAULT_LOCAL_BACKEND,
    };
  }
}

export async function parseJsonResponse<T = any>(res: Response, endpointLabel: string): Promise<T> {
  const contentType = res.headers.get('content-type') || '';
  if (!contentType.includes('application/json')) {
    const rawText = await res.text();
    const isHtml = rawText.trim().startsWith('<') || contentType.includes('text/html');
    if (isHtml) {
      throw new Error(
        `Backend endpoint '${endpointLabel}' returned an HTML page (HTTP ${res.status}) instead of JSON. Ensure your FastAPI server is running and accessible.`
      );
    }
    throw new Error(
      `Backend endpoint '${endpointLabel}' returned unexpected Content-Type '${contentType}' (HTTP ${res.status}): ${rawText.slice(0, 160)}`
    );
  }
  return res.json();
}

function getSafeInitialEndpoint(): string {
  try {
    const stored = localStorage.getItem('covenguard_api_url');
    if (stored !== null) {
      const validation = validateApiEndpoint(stored);
      if (validation.valid && stored.trim() !== '') {
        return validation.cleanUrl;
      }
      // If corrupted or invalid like "123", purge it from localStorage
      localStorage.removeItem('covenguard_api_url');
    }
  } catch {
    // Ignore storage errors
  }
  const envUrl = (import.meta as any).env.VITE_API_BASE_URL || (import.meta as any).env.COVENGUARD_API_URL;
  if (envUrl && typeof envUrl === 'string') {
    const validation = validateApiEndpoint(envUrl);
    if (validation.valid && envUrl.trim() !== '') {
      return validation.cleanUrl;
    }
  }
  return ''; // Default to same-origin relative /api proxy
}

export default function App() {
  const [apiBase, setApiBase] = useState<string>(getSafeInitialEndpoint);
  const [inputEndpoint, setInputEndpoint] = useState<string>(getSafeInitialEndpoint);
  const [endpointError, setEndpointError] = useState<string | null>(null);
  const [showConfig, setShowConfig] = useState<boolean>(false);
  const [data, setData] = useState<State | null>(null);
  const [dataPartition, setDataPartition] = useState<'iid' | 'non_iid'>('iid');
  const [attack, setAttack] = useState<string>('sign_flip');
  const [agg, setAgg] = useState<string>('median');
  const [busy, setBusy] = useState<boolean>(false);
  const [msg, setMsg] = useState<string>('Awaiting operator command');
  const [tab, setTab] = useState<string>('OVERVIEW');
  const [geminiExplanation, setGeminiExplanation] = useState<string | null>(null);
  const [geminiLoading, setGeminiLoading] = useState<boolean>(false);
  const [backendStatus, setBackendStatus] = useState<'ONLINE' | 'OFFLINE' | 'CHECKING'>('CHECKING');
  const [connectionError, setConnectionError] = useState<string | null>(null);

  // Firebase Auth & Roles
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [authToken, setAuthToken] = useState<string | null>(null);
  const [authLoading, setAuthLoading] = useState<boolean>(false);
  const [userRole, setUserRole] = useState<'admin' | 'viewer'>('viewer');
  const [firestoreConnected, setFirestoreConnected] = useState<boolean>(false);

  // Sheets modal state
  const [sheetExportUrl, setSheetExportUrl] = useState<string | null>(null);
  const [sheetConfirmOpen, setSheetConfirmOpen] = useState<boolean>(false);

  // Dataset upload progress
  const [uploadStatus, setUploadStatus] = useState<{
    uploading: boolean;
    fileName?: string;
    error?: string;
    summary?: DatasetSummary;
  }>({ uploading: false });

  const getUrl = (path: string) => {
    const base = apiBase.trim().replace(/\/$/, '');
    return `${base}${path}`;
  };

  const getHeaders = (isJson = true) => {
    const h: Record<string, string> = {};
    if (isJson) h['Content-Type'] = 'application/json';
    if (authToken) h['Authorization'] = `Bearer ${authToken}`;
    return h;
  };

  // Health and state polling with timeout and Content-Type inspection
  async function refresh() {
    const targetUrl = apiBase.trim();
    const base = targetUrl ? targetUrl.replace(/\/$/, '') : '';
    const healthUrl = `${base}/api/health`;
    const stateUrl = `${base}/api/state`;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 4000);

    try {
      // 1. Check health
      const healthRes = await fetch(healthUrl, {
        signal: controller.signal,
        headers: getHeaders(false),
      });

      if (!healthRes.ok) {
        throw new Error(`Health probe returned HTTP ${healthRes.status}: ${healthRes.statusText}`);
      }

      await parseJsonResponse(healthRes, healthUrl);

      // 2. Fetch full state
      const stateRes = await fetch(stateUrl, {
        signal: controller.signal,
        headers: getHeaders(false),
      });

      clearTimeout(timer);

      if (!stateRes.ok) {
        throw new Error(`State probe returned HTTP ${stateRes.status}: ${stateRes.statusText}`);
      }

      const stateData: State = await parseJsonResponse<State>(stateRes, stateUrl);
      setData(stateData);
      setBackendStatus('ONLINE');
      setConnectionError(null);
    } catch (err: any) {
      clearTimeout(timer);
      setBackendStatus('OFFLINE');
      const isTimeout = err.name === 'AbortError';
      const detail = isTimeout
        ? 'Connection timed out (no response within 4.0s)'
        : (err.message || 'Connection refused');
      setConnectionError(detail);
      setMsg(`FastAPI offline at ${targetUrl || 'same-origin proxy'} — ${detail}`);
    }
  }

  // Periodic polling every 5s
  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [apiBase, authToken]);

  // Firebase auth & boot verification
  useEffect(() => {
    testConnection().then((connected) => setFirestoreConnected(connected));
    const unsubscribe = initAuth(
      async (user, token) => {
        setCurrentUser(user);
        setAuthToken(token);
        const isAdmin = user.email === 'arbiyafathima975@gmail.com';
        const role = isAdmin ? 'admin' : 'viewer';
        setUserRole(role);

        try {
          await setDoc(
            doc(db, 'users', user.uid),
            {
              email: user.email,
              displayName: user.displayName || 'Operator',
              role,
              lastLogin: new Date().toISOString(),
            },
            { merge: true }
          );
        } catch (err) {
          console.warn('Firestore profile sync warning:', err);
        }
      },
      () => {
        setCurrentUser(null);
        setAuthToken(null);
        setUserRole('viewer');
      }
    );
    return () => unsubscribe();
  }, []);

  function handleSaveEndpoint() {
    const result = validateApiEndpoint(inputEndpoint);
    if (!result.valid) {
      setEndpointError(result.error || 'Invalid endpoint URL');
      return;
    }
    setEndpointError(null);
    setApiBase(result.cleanUrl);
    try {
      localStorage.setItem('covenguard_api_url', result.cleanUrl);
    } catch {
      // Ignore
    }
    setShowConfig(false);
    setBackendStatus('CHECKING');
    setMsg(`Reconnecting to FastAPI backend at ${result.cleanUrl || 'same-origin proxy'}…`);
    setTimeout(refresh, 200);
  }

  function handleSetPreset(url: string) {
    setInputEndpoint(url);
    setEndpointError(null);
  }

  async function handleGoogleLogin() {
    setAuthLoading(true);
    setMsg('Authenticating operator via Google Identity Provider…');
    try {
      const res = await googleSignIn();
      if (res) {
        setCurrentUser(res.user);
        setAuthToken(res.accessToken);
        const isAdmin = res.user.email === 'arbiyafathima975@gmail.com';
        setUserRole(isAdmin ? 'admin' : 'viewer');
        setMsg(`Authenticated as ${res.user.email} (${isAdmin ? 'ADMINISTRATOR' : 'VIEWER'})`);

        try {
          await addDoc(collection(db, 'audit_logs'), {
            event: 'OPERATOR_LOGIN',
            details: { email: res.user.email, role: isAdmin ? 'admin' : 'viewer', uid: res.user.uid },
            timestamp: new Date().toISOString(),
          });
        } catch (fsErr) {
          console.warn('Firestore audit write warning:', fsErr);
        }
      }
    } catch (err: any) {
      setMsg(`Sign in failed: ${err.message}`);
    } finally {
      setAuthLoading(false);
    }
  }

  async function handleGoogleLogout() {
    await logout();
    setCurrentUser(null);
    setAuthToken(null);
    setUserRole('viewer');
    setMsg('Signed out of Google Workspace.');
  }

  async function run() {
    if (attack !== 'none' && userRole !== 'admin') {
      setMsg('Access Denied: Byzantine attack simulations require Administrator authorization.');
      return;
    }

    setBusy(true);
    setMsg('Executing federated round on backend ML engine…');
    try {
      const r = await fetch(getUrl('/api/experiment/run'), {
        method: 'POST',
        headers: getHeaders(true),
        body: JSON.stringify({
          attack,
          aggregation: agg,
          malicious_client: 7,
          rounds: 1,
          data_partition: dataPartition,
          local_epochs: 2,
        }),
      });
      const j = await parseJsonResponse(r, '/api/experiment/run');
      if (!r.ok) throw new Error(j.detail || 'Experiment failed');
      setMsg(
        `Round ${j.round} complete · Aggregation: ${j.aggregation.toUpperCase()} · ${j.flagged_clients.length} node(s) quarantined`
      );
      setGeminiExplanation(null);
      await refresh();

      if (currentUser) {
        try {
          await addDoc(collection(db, 'experiments'), {
            round: j.round,
            attack: j.attack,
            aggregation: j.aggregation,
            flaggedClients: j.flagged_clients.join(','),
            testAccuracy: j.test.accuracy,
            testF1: j.test.f1,
            executedBy: currentUser.email || currentUser.uid,
            timestamp: new Date().toISOString(),
          });
        } catch (fsErr) {
          handleFirestoreError(fsErr, OperationType.WRITE, 'experiments');
        }
      }
    } catch (e: any) {
      setMsg(`Execution failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function upload(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;

    if (!f.name.toLowerCase().endsWith('.npz')) {
      setUploadStatus({
        uploading: false,
        error: 'Invalid file extension. Please select an official .npz MedMNIST archive.',
      });
      return;
    }

    const fd = new FormData();
    fd.append('file', f);
    setBusy(true);
    setUploadStatus({ uploading: true, fileName: f.name });
    setMsg(`Uploading and validating ${f.name} (train/val/test partitions)…`);

    try {
      const r = await fetch(getUrl('/api/dataset/upload'), {
        method: 'POST',
        headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
        body: fd,
      });
      const j = await parseJsonResponse(r, '/api/dataset/upload');
      if (!r.ok) throw new Error(j.detail || 'Upload failed');
      setUploadStatus({ uploading: false, fileName: f.name, summary: j });
      setMsg(`Loaded ${j.name}: ${j.train_samples} training samples partitioned across 10 hospitals.`);
      await refresh();
    } catch (err: any) {
      setUploadStatus({ uploading: false, fileName: f.name, error: err.message });
      setMsg(`Upload error: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  // Google Picker flow
  async function handleOpenDrivePicker() {
    if (!currentUser) {
      await handleGoogleLogin();
      return;
    }
    setMsg('Opening Google Picker to select dataset from Google Drive…');
    try {
      await openGooglePicker(async (file) => {
        setMsg(`Downloading ${file.name} from Google Drive via Bearer token…`);
        setBusy(true);
        setUploadStatus({ uploading: true, fileName: file.name });
        try {
          const blob = await downloadDriveFile(file.id);
          const fd = new FormData();
          fd.append('file', blob, file.name);
          const r = await fetch(getUrl('/api/dataset/upload'), {
            method: 'POST',
            headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
            body: fd,
          });
          const j = await parseJsonResponse(r, '/api/dataset/upload');
          if (!r.ok) throw new Error(j.detail || 'Failed processing file from Google Drive');
          setUploadStatus({ uploading: false, fileName: file.name, summary: j });
          setMsg(`Imported from Drive: ${j.name} (${j.train_samples} samples)`);
          await refresh();
        } catch (pickerUploadErr: any) {
          setUploadStatus({ uploading: false, fileName: file.name, error: pickerUploadErr.message });
          setMsg(`Drive import error: ${pickerUploadErr.message}`);
        } finally {
          setBusy(false);
        }
      });
    } catch (pickerErr: any) {
      setMsg(`Picker error: ${pickerErr.message}`);
    }
  }

  // Google Sheets Export flow
  async function handleConfirmSheetsExport() {
    setSheetConfirmOpen(false);
    if (!last) return;
    setBusy(true);
    setMsg('Exporting federated round telemetry to Google Sheets…');
    try {
      const res = await exportToGoogleSheets({
        round: last.round,
        attack: last.attack,
        aggregation: last.aggregation,
        accuracy: last.test.accuracy,
        precision: last.test.precision,
        recall: last.test.recall,
        f1: last.test.f1,
        clients: last.client_evidence,
        events: data?.events || [],
      });
      setSheetExportUrl(res.spreadsheetUrl);
      setMsg(`Exported to Google Sheets: ${res.spreadsheetUrl}`);
    } catch (sheetErr: any) {
      setMsg(`Google Sheets export error: ${sheetErr.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function reset() {
    if (userRole !== 'admin') {
      setMsg('Access Denied: Network reset requires Administrator authorization.');
      return;
    }
    setBusy(true);
    try {
      const r = await fetch(getUrl('/api/reset'), {
        method: 'POST',
        headers: getHeaders(true),
      });
      const j = await parseJsonResponse(r, '/api/reset');
      if (!r.ok) {
        throw new Error(j.detail || 'Reset failed');
      }
      setMsg('Hospital network telemetry reset to baseline state.');
      setGeminiExplanation(null);
      await refresh();
    } catch (err: any) {
      setMsg(`Reset failed: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function requestGeminiAnalysis() {
    if (!last) return;
    setGeminiLoading(true);
    setMsg('Querying Gemini Security Analyst for incident explanation…');
    try {
      const payload = {
        round: last.round,
        attack_type: last.attack,
        aggregation: last.aggregation,
        quarantined_nodes: last.flagged_clients,
        client_evidence: last.client_evidence.filter(
          (c) => c.status === 'QUARANTINED' || c.id === last.malicious_client
        ),
        metrics: last.test,
      };

      const res = await fetch(getUrl('/api/ai/explain'), {
        method: 'POST',
        headers: getHeaders(true),
        body: JSON.stringify({ evidence: payload }),
      });
      const resData = await parseJsonResponse(res, '/api/ai/explain');
      if (!res.ok) throw new Error(resData.detail || 'Failed to generate explanation');
      setGeminiExplanation(resData.explanation);
      setMsg('Gemini Security Analyst incident briefing received.');
    } catch (err: any) {
      setGeminiExplanation(`Security Analysis Unavailable: ${err.message}`);
    } finally {
      setGeminiLoading(false);
    }
  }

  const clients: Client[] =
    backendStatus === 'ONLINE' && data?.clients?.length
      ? data.clients
      : [];

  const last = data?.last_experiment;

  const nav = [
    ['OVERVIEW', Activity],
    ['HOSPITAL GRID', Network],
    ['THREAT LAB', FlaskConical],
    ['MODEL INTEGRITY', BrainCircuit],
    ['AUDIT LOG', Terminal],
  ] as const;

  return (
    <div className="shell">
      <div className="scanline" />
      <aside className="sidebar">
        <div className="brand">
          <div className="brandmark">
            <Shield size={25} />
          </div>
          <div>
            <b>COVENGUARD</b>
            <small>AI DEFENCE NETWORK</small>
          </div>
        </div>
        <div className="side-label">OPERATIONS</div>
        {nav.map(([n, Icon]) => (
          <button
            className={'nav ' + (tab === n ? 'active' : '')}
            onClick={() => setTab(n)}
            key={n}
          >
            <Icon size={17} />
            {n}
            <ChevronRight size={14} className="navchev" />
          </button>
        ))}

        <div className="side-bottom">
          {/* Operator Auth Card */}
          {currentUser ? (
            <div
              style={{
                background: '#0a1d29',
                border: userRole === 'admin' ? '1px solid #56e4bd' : '1px solid #1c3d4e',
                padding: '8px 10px',
                marginBottom: 6,
                borderRadius: 3,
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 7,
                  fontSize: 10,
                  color: userRole === 'admin' ? '#56e4bd' : '#6be5c9',
                  fontFamily: 'IBM Plex Mono',
                }}
              >
                <UserIcon size={13} />
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {currentUser.displayName || currentUser.email}
                </span>
              </div>
              <div
                style={{
                  fontSize: 9,
                  fontFamily: 'IBM Plex Mono',
                  color: userRole === 'admin' ? '#56e4bd' : '#91a7b4',
                  marginTop: 3,
                  fontWeight: 600,
                }}
              >
                ROLE: {userRole.toUpperCase()}
              </div>
              <button
                className="secondary"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 5,
                  fontSize: 9,
                  marginTop: 6,
                  padding: '4px 6px',
                  width: '100%',
                }}
                onClick={handleGoogleLogout}
              >
                <LogOut size={11} /> SIGN OUT
              </button>
            </div>
          ) : (
            <button
              onClick={handleGoogleLogin}
              disabled={authLoading}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                background: '#ffffff',
                color: '#1f1f1f',
                border: '1px solid #747775',
                borderRadius: 4,
                padding: '8px 12px',
                fontFamily: 'Roboto, sans-serif',
                fontSize: 11,
                fontWeight: 500,
                cursor: 'pointer',
                marginBottom: 8,
              }}
            >
              <svg width="16" height="16" viewBox="0 0 48 48">
                <path
                  fill="#EA4335"
                  d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"
                />
                <path
                  fill="#4285F4"
                  d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"
                />
                <path
                  fill="#FBBC05"
                  d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"
                />
                <path
                  fill="#34A853"
                  d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"
                />
              </svg>
              <span>{authLoading ? 'Signing in…' : 'Sign in with Google'}</span>
            </button>
          )}

          <button
            className="secondary"
            style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 10 }}
            onClick={() => {
              setInputEndpoint(apiBase);
              setEndpointError(null);
              setShowConfig(!showConfig);
            }}
          >
            <Settings size={13} />
            <span>CONFIGURE ENDPOINT</span>
          </button>
          <div className="connection">
            <span
              className="pulse"
              style={{
                background: backendStatus === 'ONLINE' ? '#56e4bd' : '#ff5d69',
                boxShadow:
                  backendStatus === 'ONLINE'
                    ? '0 0 12px #56e4bd'
                    : '0 0 12px #ff5d69',
              }}
            />{' '}
            {backendStatus === 'ONLINE'
              ? `API ONLINE (${data?.persistence?.toUpperCase() || 'SQLITE'})`
              : backendStatus === 'CHECKING'
              ? 'VERIFYING ENDPOINT…'
              : 'BACKEND OFFLINE'}
          </div>
          <small>
            FIRESTORE: {firestoreConnected ? 'SYNCED' : 'OFFLINE'} · RBAC ACTIVE
          </small>
        </div>
      </aside>

      <main className="main">
        <header>
          <div>
            <div className="eyebrow">
              FEDERATED HEALTHCARE SECURITY / BYZANTINE ROBUST AGGREGATION
            </div>
            <h1>
              {tab === 'OVERVIEW'
                ? 'Command overview'
                : tab === 'HOSPITAL GRID'
                ? 'Hospital network'
                : tab === 'THREAT LAB'
                ? 'Threat simulation lab'
                : tab === 'MODEL INTEGRITY'
                ? 'Model integrity'
                : 'Security audit log'}
            </h1>
          </div>
          <div className="topright">
            <span className="live">
              <i
                style={{
                  background:
                    backendStatus === 'ONLINE'
                      ? '#56e4bd'
                      : backendStatus === 'CHECKING'
                      ? '#ffd166'
                      : '#ff5d69',
                }}
              />{' '}
              SYSTEM {backendStatus}
            </span>
            <button
              className="iconbtn"
              onClick={refresh}
              title="Refresh Telemetry & Reconnect"
              style={{ display: 'flex', alignItems: 'center', gap: 4 }}
            >
              <RefreshCw size={14} />
            </button>
          </div>
        </header>

        <div className="statusbar">
          <span>
            <RadioTower size={14} /> NETWORK STATUS:{' '}
            <b>{backendStatus === 'ONLINE' ? 'MONITORED' : 'DISCONNECTED'}</b>
          </span>
          <span>
            ENDPOINT: <b>{apiBase || 'Same-origin /api Proxy'}</b>
          </span>
          <span>
            FEDERATED ROUND <b>{data?.round || 0}</b>
          </span>
          <span>
            DATA SOURCE <b>{data?.dataset?.name || 'INITIALIZING'}</b>
          </span>
          <span>
            PERSISTENCE <b>{data?.persistence ? data.persistence.toUpperCase() : 'SQLITE'}</b>
          </span>
        </div>

        {/* Clear Offline Warning Banner */}
        {backendStatus === 'OFFLINE' && (
          <div
            style={{
              margin: '14px 0',
              padding: '12px 18px',
              background: 'rgba(38, 12, 18, 0.96)',
              border: '1px solid #ff5d69',
              borderRadius: 3,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 14,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <Server size={20} color="#ff5d69" />
              <div>
                <b style={{ color: '#ff818b', fontSize: 13, fontFamily: 'IBM Plex Mono', letterSpacing: 0.5 }}>
                  FASTAPI BACKEND OFFLINE
                </b>
                <p style={{ margin: '3px 0 0', fontSize: 11, color: '#e0c2c6', lineHeight: 1.4 }}>
                  Configured endpoint: <code>{apiBase || 'http://localhost:8000'}</code>
                  {connectionError && (
                    <span style={{ color: '#ffb3ba', marginLeft: 8 }}>
                      ({connectionError})
                    </span>
                  )}
                </p>
                <small style={{ color: '#9d7c82', fontSize: 9, fontFamily: 'IBM Plex Mono' }}>
                  Ensure FastAPI is started locally via <code>uvicorn app.main:app --port 8000</code> or deploy to Google Cloud Run.
                </small>
              </div>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button
                className="secondary"
                style={{ padding: '7px 12px', fontSize: 10, display: 'flex', alignItems: 'center', gap: 5 }}
                onClick={refresh}
              >
                <RefreshCw size={12} /> RECONNECT
              </button>
              <button
                className="primary"
                style={{ margin: 0, padding: '7px 12px', fontSize: 10 }}
                onClick={() => {
                  setInputEndpoint(apiBase);
                  setEndpointError(null);
                  setShowConfig(true);
                }}
              >
                CONFIGURE
              </button>
            </div>
          </div>
        )}

        {/* Backend Configuration Panel with Strict Validation */}
        {showConfig && (
          <div
            className="panel"
            style={{
              margin: '14px 0',
              border: '1px solid #4dddc5',
              background: 'rgba(8, 26, 36, 0.98)',
            }}
          >
            <PanelTitle title="FASTAPI BACKEND CONNECTION SETTINGS" tag="ENDPOINT VALIDATION" />
            <p className="muted">
              Configure the target HTTP/HTTPS endpoint for the Python ML FastAPI backend.
              Numeric values like <code>123</code> are rejected.
            </p>

            <div style={{ margin: '10px 0' }}>
              <label style={{ fontSize: 10, color: '#7e98a5', fontFamily: 'IBM Plex Mono', display: 'block', marginBottom: 6 }}>
                FASTAPI BASE URL
              </label>
              <input
                type="text"
                value={inputEndpoint}
                placeholder="http://localhost:8000 or https://covenguard-backend-...a.run.app"
                onChange={(e) => {
                  setInputEndpoint(e.target.value);
                  const v = validateApiEndpoint(e.target.value);
                  if (!v.valid) {
                    setEndpointError(v.error || null);
                  } else {
                    setEndpointError(null);
                  }
                }}
                style={{
                  width: '100%',
                  background: '#06121c',
                  border: endpointError ? '1px solid #ff5d69' : '1px solid #294656',
                  color: endpointError ? '#ff818b' : '#65dfbd',
                  padding: '9px 12px',
                  fontFamily: 'IBM Plex Mono',
                  fontSize: 12,
                }}
              />
              {endpointError && (
                <div style={{ color: '#ff6974', fontSize: 10, fontFamily: 'IBM Plex Mono', marginTop: 5, display: 'flex', alignItems: 'center', gap: 6 }}>
                  <AlertOctagon size={13} /> {endpointError}
                </div>
              )}
            </div>

            {/* Presets */}
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 14 }}>
              <button
                className="secondary"
                style={{ fontSize: 9, padding: '4px 8px' }}
                onClick={() => handleSetPreset('')}
              >
                SAME-ORIGIN PROXY (/api)
              </button>
              <button
                className="secondary"
                style={{ fontSize: 9, padding: '4px 8px' }}
                onClick={() => handleSetPreset('http://localhost:8000')}
              >
                STANDALONE LOCAL (http://localhost:8000)
              </button>
              <button
                className="secondary"
                style={{ fontSize: 9, padding: '4px 8px' }}
                onClick={() => handleSetPreset('http://localhost:8001')}
              >
                LOCAL DEV PORT (http://localhost:8001)
              </button>
            </div>

            <div style={{ display: 'flex', gap: 10 }}>
              <button
                className="primary"
                style={{ margin: 0 }}
                onClick={handleSaveEndpoint}
                disabled={Boolean(endpointError)}
              >
                SAVE & RECONNECT
              </button>
              <button
                className="secondary"
                style={{ margin: 0 }}
                onClick={() => {
                  setInputEndpoint(apiBase);
                  setEndpointError(null);
                  setShowConfig(false);
                }}
              >
                CANCEL
              </button>
            </div>
          </div>
        )}

        {/* User Confirmation Dialog for Mutating Workspace Operations */}
        {sheetConfirmOpen && (
          <div
            className="panel"
            style={{
              margin: '14px 0',
              border: '1px solid #56e4bd',
              background: 'rgba(12, 38, 48, 0.98)',
            }}
          >
            <PanelTitle title="CONFIRM GOOGLE SHEETS EXPORT" tag="WORKSPACE API" />
            <p className="muted">
              You are about to create a new Google Spreadsheet in your Google Drive named{' '}
              <b>
                CovenGuard AI - Round {last?.round} Telemetry Audit
              </b>{' '}
              and write 3 worksheets (Executive Summary, Hospital Trust Ledger, Security Audit Journal).
            </p>
            <div style={{ display: 'flex', gap: 12, marginTop: 14 }}>
              <button className="primary" onClick={handleConfirmSheetsExport}>
                CONFIRM & GENERATE SPREADSHEET
              </button>
              <button className="secondary" onClick={() => setSheetConfirmOpen(false)}>
                CANCEL
              </button>
            </div>
          </div>
        )}

        {/* TAB 1: OVERVIEW */}
        {tab === 'OVERVIEW' && (
          <>
            <section className="hero">
              <div>
                <div className="eyebrow cyan">ZERO-TRUST FEDERATED LEARNING</div>
                <h2>
                  Protect the model.
                  <br />
                  <em>Preserve the network.</em>
                </h2>
                <p>
                  10 independent simulated hospital nodes train a collaborative AI model.
                  Inspect local model update weights, detect anomalous distributions, isolate
                  Byzantine actors, and aggregate through robust coordinate-wise median or
                  trimmed mean.
                </p>
                <div style={{ display: 'flex', gap: 10, marginTop: 14 }}>
                  <button className="primary" onClick={() => setTab('THREAT LAB')}>
                    OPEN THREAT LAB <ChevronRight size={16} />
                  </button>
                  <button
                    className="secondary"
                    style={{ display: 'flex', alignItems: 'center', gap: 7 }}
                    onClick={handleOpenDrivePicker}
                  >
                    <FolderOpen size={15} /> PICK DATASET FROM DRIVE
                  </button>
                </div>
              </div>
              <div className="radar">
                <div className="radar-ring r1" />
                <div className="radar-ring r2" />
                <div className="radar-ring r3" />
                <div className="radar-sweep" />
                <div className="radar-center">
                  <Shield size={34} />
                  <small>GUARD</small>
                </div>
                {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((n) => (
                  <span
                    key={n}
                    className={
                      'dot d' +
                      n +
                      (clients[n]?.status === 'QUARANTINED' ? ' danger' : '')
                    }
                  />
                ))}
              </div>
            </section>

            <section className="metrics">
              <Metric
                label="CONNECTED CLIENTS"
                value={backendStatus === 'ONLINE' ? clients.length : 0}
                suffix={backendStatus === 'ONLINE' ? '/ 10' : ' (OFFLINE)'}
                icon={<Network />}
              />
              <Metric
                label="HEALTHY NODES"
                value={backendStatus === 'ONLINE' ? clients.filter((c) => c.status === 'HEALTHY').length : 0}
                icon={<CheckCircle2 />}
              />
              <Metric
                label="QUARANTINED"
                value={backendStatus === 'ONLINE' ? clients.filter((c) => c.status === 'QUARANTINED').length : 0}
                icon={<AlertTriangle />}
                danger={backendStatus === 'ONLINE' && clients.filter((c) => c.status === 'QUARANTINED').length > 0}
              />
              <Metric
                label="GLOBAL TEST ACCURACY"
                value={backendStatus === 'ONLINE' && last ? `${(last.test.accuracy * 100).toFixed(1)}%` : '—'}
                icon={<Activity />}
              />
            </section>

            <section className="grid2">
              <div className="panel">
                <PanelTitle title="NETWORK TOPOLOGY" tag={backendStatus === 'ONLINE' ? `${clients.length} CLIENTS` : 'DISCONNECTED'} />
                {backendStatus === 'ONLINE' && clients.length > 0 ? (
                  <div className="nodes">
                    {clients.map((c) => (
                      <div
                        className={'node ' + (c.status === 'QUARANTINED' ? 'quarantined' : '')}
                        key={c.id}
                      >
                        <div className="nodeicon">
                          {c.status === 'QUARANTINED' ? (
                            <ShieldAlert size={17} />
                          ) : (
                            <Network size={17} />
                          )}
                        </div>
                        <b>H-{String(c.id).padStart(2, '0')}</b>
                        <small>{c.status === 'QUARANTINED' ? 'QUARANTINED' : 'ONLINE'}</small>
                        <div className="trust">
                          <i
                            style={{
                              width: `${c.trust}%`,
                              background: c.status === 'QUARANTINED' ? '#ff5d69' : '#59dfc0',
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ padding: '36px 20px', textAlign: 'center', background: 'rgba(10, 16, 24, 0.6)', border: '1px dashed #3a2228', borderRadius: 4 }}>
                    <Server size={28} color="#ff5d69" style={{ margin: '0 auto 10px', display: 'block' }} />
                    <b style={{ color: '#ff818b', fontSize: 13, fontFamily: 'IBM Plex Mono' }}>FASTAPI BACKEND DISCONNECTED</b>
                    <p style={{ fontSize: 11, color: '#8fa0aa', margin: '6px 0 0' }}>
                      Hospital node telemetry stream is inactive. Reconnect to FastAPI to monitor live nodes.
                    </p>
                  </div>
                )}
              </div>

              <div className="panel">
                <PanelTitle title="MODEL PERFORMANCE" tag="HELD-OUT TEST SET" />
                {last ? (
                  <>
                    <div className="bigmetric">
                      {(last.test.accuracy * 100).toFixed(1)}
                      <small>%</small>
                    </div>
                    <div className="metric-caption">
                      TEST ACCURACY · ROUND {last.round} · {last.aggregation.toUpperCase()}
                    </div>
                    <div className="perfgrid">
                      <span>
                        PRECISION <b>{(last.test.precision * 100).toFixed(1)}%</b>
                      </span>
                      <span>
                        RECALL <b>{(last.test.recall * 100).toFixed(1)}%</b>
                      </span>
                      <span>
                        F1 SCORE <b>{(last.test.f1 * 100).toFixed(1)}%</b>
                      </span>
                    </div>
                  </>
                ) : (
                  <div className="empty">
                    Run an experiment from the Threat Lab to populate measured model metrics.
                  </div>
                )}
              </div>
            </section>
          </>
        )}

        {/* TAB 2: HOSPITAL GRID */}
        {tab === 'HOSPITAL GRID' && (
          <section className="panel">
            <PanelTitle title="CLIENT REGISTRY" tag={backendStatus === 'ONLINE' ? 'LIVE TELEMETRY' : 'DISCONNECTED'} />
            {backendStatus === 'ONLINE' && clients.length > 0 ? (
              <div className="clienttable">
                <div className="tr th">
                  <span>CLIENT</span>
                  <span>STATE</span>
                  <span>TRUST</span>
                  <span>UPDATE NORM (L2)</span>
                  <span>MEAN COSINE</span>
                </div>
                {clients.map((c) => (
                  <div className="tr" key={c.id}>
                    <b>HOSPITAL {String(c.id).padStart(2, '0')}</b>
                    <span className={c.status === 'QUARANTINED' ? 'red' : 'green'}>
                      {c.status}
                    </span>
                    <span>{c.trust}%</span>
                    <span>{c.l2_norm !== undefined ? c.l2_norm : '—'}</span>
                    <span>{c.mean_cosine !== undefined ? c.mean_cosine : '—'}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ padding: '40px 20px', textAlign: 'center', background: 'rgba(10, 16, 24, 0.6)', border: '1px dashed #3a2228', margin: '14px 0', borderRadius: 4 }}>
                <RadioTower size={32} color="#ff5d69" style={{ margin: '0 auto 12px', display: 'block' }} />
                <b style={{ color: '#ff818b', fontSize: 13, fontFamily: 'IBM Plex Mono' }}>NO CLIENT TELEMETRY AVAILABLE</b>
                <p style={{ fontSize: 11, color: '#8fa0aa', margin: '6px 0 0' }}>
                  FastAPI backend is offline. Start FastAPI service or update endpoint in settings to stream hospital records.
                </p>
              </div>
            )}
          </section>
        )}

        {/* TAB 3: THREAT LAB */}
        {tab === 'THREAT LAB' && (
          <section className="grid2">
            <div className="panel">
              <PanelTitle title="ATTACK CONFIGURATION" tag="CONTROLLED TEST" />
              <p className="muted">
                Inject a controlled malicious model update into one simulated client. Hospital
                07 is designated as the target adversarial node.
              </p>

              {userRole !== 'admin' && (
                <div
                  style={{
                    background: '#1a1924',
                    border: '1px solid #784860',
                    padding: '8px 12px',
                    marginBottom: 14,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    fontSize: 11,
                    color: '#e4aab8',
                  }}
                >
                  <Lock size={14} color="#ff818b" />
                  <span>
                    Administrator authorization required to execute Byzantine attacks. Viewer mode active.
                  </span>
                </div>
              )}

              <label>ATTACK VECTOR</label>
              <select value={attack} onChange={(e) => setAttack(e.target.value)}>
                <option value="sign_flip">Sign-flipping / amplified gradient poison</option>
                <option value="noise">Random Gaussian noise injection</option>
                <option value="none">No attack / honest baseline</option>
              </select>

              <label>AGGREGATION STRATEGY</label>
              <select value={agg} onChange={(e) => setAgg(e.target.value)}>
                <option value="median">Coordinate-wise median (Byzantine-robust)</option>
                <option value="trimmed_mean">Trimmed mean (Byzantine-robust)</option>
                <option value="fedavg">Standard FedAvg (Vulnerable to outliers)</option>
              </select>

              <label>DATA DISTRIBUTION PARTITION</label>
              <select value={dataPartition} onChange={(e) => setDataPartition(e.target.value as 'iid' | 'non_iid')}>
                <option value="iid">IID: Uniform patient distribution across 10 hospitals</option>
                <option value="non_iid">Non-IID: Dirichlet demographic skew (clinical disparity)</option>
              </select>

              <div className="target">
                <AlertTriangle size={17} />
                <span>
                  TARGET NODE <b>HOSPITAL 07</b>
                </span>
                <span className="tag-red">SIMULATED ADVERSARY</span>
              </div>

              <button
                className="primary full"
                disabled={busy || backendStatus !== 'ONLINE' || (attack !== 'none' && userRole !== 'admin')}
                onClick={run}
              >
                {backendStatus !== 'ONLINE'
                  ? 'FASTAPI BACKEND REQUIRED (OFFLINE)'
                  : busy
                  ? 'PROCESSING ROUND…'
                  : 'EXECUTE FEDERATED ROUND'}{' '}
                <ChevronRight size={16} />
              </button>

              <button
                className="secondary full"
                onClick={reset}
                disabled={busy || backendStatus !== 'ONLINE' || userRole !== 'admin'}
              >
                RESET NETWORK TO BASELINE
              </button>

              <div className="console">
                <span>● OPERATOR CONSOLE</span>
                <p>{msg}</p>
              </div>

              {last && (
                <div style={{ marginTop: 15 }}>
                  <button
                    className="primary full"
                    style={{
                      background: '#153f47',
                      color: '#65dfbd',
                      border: '1px solid #285f67',
                    }}
                    onClick={requestGeminiAnalysis}
                    disabled={geminiLoading}
                  >
                    <Bot size={16} />
                    {geminiLoading
                      ? 'ANALYZING EVIDENCE WITH GEMINI…'
                      : 'REQUEST GEMINI SECURITY INCIDENT EXPLANATION'}
                  </button>
                </div>
              )}
            </div>

            <div className="panel">
              <PanelTitle title="DETECTION PIPELINE" tag="MULTI-METRIC" />
              {[
                'Payload finite-value validation',
                'L2 update norm deviation analysis',
                'Pairwise cosine similarity alignment',
                'Median Absolute Deviation (MAD) robust score',
                'Dynamic trust ledger update & quarantine',
                'Robust geometric aggregation',
              ].map((s, i) => (
                <div className="pipeline" key={s}>
                  <span>{String(i + 1).padStart(2, '0')}</span>
                  <b>{s}</b>
                  <CheckCircle2 size={15} />
                </div>
              ))}

              {geminiExplanation && (
                <div
                  style={{
                    marginTop: 20,
                    padding: 14,
                    background: '#091c28',
                    border: '1px solid #2f5d6e',
                    borderRadius: 4,
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      color: '#6df0d4',
                      fontFamily: 'IBM Plex Mono',
                      fontSize: 11,
                      fontWeight: 600,
                      marginBottom: 8,
                    }}
                  >
                    <Sparkles size={15} /> GEMINI SECURITY ANALYST BRIEFING
                  </div>
                  <pre
                    style={{
                      whiteSpace: 'pre-wrap',
                      fontFamily: 'IBM Plex Mono',
                      fontSize: 10,
                      color: '#bad2dc',
                      lineHeight: 1.6,
                      maxHeight: 220,
                      overflowY: 'auto',
                    }}
                  >
                    {geminiExplanation}
                  </pre>
                </div>
              )}

              <p className="footnote">
                Detection uses statistical and geometric heuristics. Always cross-validate
                against legitimate non-IID patient distributions to mitigate false positives
                in clinical research networks.
              </p>
            </div>
          </section>
        )}

        {/* TAB 4: MODEL INTEGRITY */}
        {tab === 'MODEL INTEGRITY' && (
          <>
            {/* Model Performance Comparison Cards */}
            {last && (
              <section className="panel" style={{ marginBottom: 16 }}>
                <PanelTitle title="EXPERIMENTAL PERFORMANCE COMPARISON" tag="MEASURED BENCHMARK RESULTS" />
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 14, marginTop: 12 }}>
                  {/* Baseline Pre-Round */}
                  <div style={{ background: '#091520', border: '1px solid #1e3545', padding: '12px 16px', borderRadius: 4 }}>
                    <div style={{ fontSize: 10, color: '#7e95a6', fontFamily: 'IBM Plex Mono', marginBottom: 4 }}>
                      1. BASELINE ACCURACY (PRE-ROUND)
                    </div>
                    <div style={{ fontSize: 24, fontWeight: 700, color: '#9db4c0', fontFamily: 'IBM Plex Mono' }}>
                      {last.baseline?.accuracy !== undefined ? `${(last.baseline.accuracy * 100).toFixed(1)}%` : '—'}
                    </div>
                    <div style={{ fontSize: 11, color: '#688291', marginTop: 4 }}>
                      Loss: {last.baseline?.loss ?? '—'} · F1: {last.baseline?.f1 !== undefined ? (last.baseline.f1 * 100).toFixed(1) + '%' : '—'}
                    </div>
                    <small style={{ fontSize: 9, color: '#526b79', display: 'block', marginTop: 6 }}>
                      Evaluated on held-out test split prior to current aggregation
                    </small>
                  </div>

                  {/* Poisoned Naive FedAvg */}
                  <div style={{ background: '#1c1015', border: '1px solid #58242e', padding: '12px 16px', borderRadius: 4 }}>
                    <div style={{ fontSize: 10, color: '#e4959f', fontFamily: 'IBM Plex Mono', marginBottom: 4 }}>
                      2. ACCURACY AFTER POISONING (NAIVE FEDAVG)
                    </div>
                    <div style={{ fontSize: 24, fontWeight: 700, color: '#ff818b', fontFamily: 'IBM Plex Mono' }}>
                      {last.poisoned_fedavg?.accuracy !== undefined ? `${(last.poisoned_fedavg.accuracy * 100).toFixed(1)}%` : '—'}
                    </div>
                    <div style={{ fontSize: 11, color: '#d0808b', marginTop: 4 }}>
                      Loss: {last.poisoned_fedavg?.loss ?? '—'} · Attack: {last.attack.toUpperCase()}
                    </div>
                    <small style={{ fontSize: 9, color: '#9a535f', display: 'block', marginTop: 6 }}>
                      {last.attack !== 'none'
                        ? 'Degradation under unconstrained Byzantine adversary'
                        : 'Clean baseline run (no poison injected)'}
                    </small>
                  </div>

                  {/* Robust Aggregation */}
                  <div style={{ background: '#092120', border: '1px solid #235850', padding: '12px 16px', borderRadius: 4 }}>
                    <div style={{ fontSize: 10, color: '#62dcbf', fontFamily: 'IBM Plex Mono', marginBottom: 4 }}>
                      3. ACCURACY AFTER ROBUST AGGREGATION
                    </div>
                    <div style={{ fontSize: 24, fontWeight: 700, color: '#56e4bd', fontFamily: 'IBM Plex Mono' }}>
                      {(last.test.accuracy * 100).toFixed(1)}%
                    </div>
                    <div style={{ fontSize: 11, color: '#8de8ce', marginTop: 4 }}>
                      F1: {(last.test.f1 * 100).toFixed(1)}% · Strategy: {last.aggregation.toUpperCase()}
                    </div>
                    <small style={{ fontSize: 9, color: '#4a9c87', display: 'block', marginTop: 6 }}>
                      Multi-metric anomaly quarantine + {last.aggregation.toUpperCase()}
                    </small>
                  </div>
                </div>
              </section>
            )}

            <section className="grid2">
              <div className="panel">
                <PanelTitle title="CLASSIFICATION METRICS" tag="MEASURED ON HELD-OUT TEST SPLIT" />
                {last ? (
                  <>
                    <div className="chart">
                      <ResponsiveContainer width="100%" height={210}>
                        <BarChart
                          data={[
                            { name: 'Accuracy', value: last.test.accuracy * 100 },
                            { name: 'Precision', value: last.test.precision * 100 },
                            { name: 'Recall', value: last.test.recall * 100 },
                            { name: 'F1 Score', value: last.test.f1 * 100 },
                          ]}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#203443" />
                          <XAxis dataKey="name" stroke="#7e95a6" />
                          <YAxis domain={[0, 100]} stroke="#7e95a6" />
                          <Tooltip
                            contentStyle={{
                              background: '#0b1823',
                              border: '1px solid #294354',
                            }}
                          />
                          <Bar dataKey="value" fill="#53e0c1" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Clinical Confusion Matrix */}
                    <div style={{ marginTop: 18 }}>
                      <div style={{ fontSize: 10, color: '#7e95a6', fontFamily: 'IBM Plex Mono', marginBottom: 8 }}>
                        CLINICAL CONFUSION MATRIX (TEST SPLIT)
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
                        <div style={{ background: '#0b1f20', border: '1px solid #265952', padding: '10px 12px', borderRadius: 4 }}>
                          <span style={{ fontSize: 9, color: '#68caa8', display: 'block' }}>TRUE POSITIVE (TP)</span>
                          <b style={{ fontSize: 16, color: '#56e4bd', fontFamily: 'IBM Plex Mono' }}>
                            {last.test.confusion?.tp ?? '—'}
                          </b>
                          <small style={{ fontSize: 9, color: '#578e7c', display: 'block' }}>Pneumonia correctly flagged</small>
                        </div>
                        <div style={{ background: '#1c151a', border: '1px solid #57293b', padding: '10px 12px', borderRadius: 4 }}>
                          <span style={{ fontSize: 9, color: '#d0869a', display: 'block' }}>FALSE POSITIVE (FP)</span>
                          <b style={{ fontSize: 16, color: '#ff818b', fontFamily: 'IBM Plex Mono' }}>
                            {last.test.confusion?.fp ?? '—'}
                          </b>
                          <small style={{ fontSize: 9, color: '#8d4f5f', display: 'block' }}>Type I false alarm</small>
                        </div>
                        <div style={{ background: '#1c151a', border: '1px solid #57293b', padding: '10px 12px', borderRadius: 4 }}>
                          <span style={{ fontSize: 9, color: '#d0869a', display: 'block' }}>FALSE NEGATIVE (FN)</span>
                          <b style={{ fontSize: 16, color: '#ff818b', fontFamily: 'IBM Plex Mono' }}>
                            {last.test.confusion?.fn ?? '—'}
                          </b>
                          <small style={{ fontSize: 9, color: '#8d4f5f', display: 'block' }}>Type II missed diagnosis</small>
                        </div>
                        <div style={{ background: '#0b1f20', border: '1px solid #265952', padding: '10px 12px', borderRadius: 4 }}>
                          <span style={{ fontSize: 9, color: '#68caa8', display: 'block' }}>TRUE NEGATIVE (TN)</span>
                          <b style={{ fontSize: 16, color: '#56e4bd', fontFamily: 'IBM Plex Mono' }}>
                            {last.test.confusion?.tn ?? '—'}
                          </b>
                          <small style={{ fontSize: 9, color: '#578e7c', display: 'block' }}>Normal correctly identified</small>
                        </div>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#7a929f', marginTop: 8, fontFamily: 'IBM Plex Mono' }}>
                        <span>False Positive Rate (FPR): <b>{last.test.fpr !== undefined ? `${(last.test.fpr * 100).toFixed(1)}%` : '—'}</b></span>
                        <span>Evaluation Loss: <b>{last.test.loss ?? '—'}</b></span>
                      </div>
                    </div>

                    <div style={{ marginTop: 15 }}>
                      <button
                        className="primary"
                        style={{ display: 'flex', alignItems: 'center', gap: 8 }}
                        onClick={() => {
                          if (!currentUser) {
                            handleGoogleLogin();
                          } else {
                            setSheetConfirmOpen(true);
                          }
                        }}
                      >
                        <FileSpreadsheet size={15} /> EXPORT RESULTS TO GOOGLE SHEETS
                      </button>
                    </div>
                  </>
                ) : (
                  <div className="empty">No model evaluation recorded yet. Execute a round from Threat Lab.</div>
                )}
              </div>

              <div className="panel">
                <PanelTitle title="LOSS CONVERGENCE & AGGREGATION" tag="TRAIN / VAL / TEST LOSS" />
                {last ? (
                  <>
                    {/* Multi-round Loss Curves */}
                    {last.loss_history && last.loss_history.length > 0 ? (
                      <div className="chart" style={{ marginBottom: 14 }}>
                        <ResponsiveContainer width="100%" height={210}>
                          <LineChart data={last.loss_history}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#203443" />
                            <XAxis dataKey="round" stroke="#7e95a6" label={{ value: 'Round', position: 'insideBottomRight', offset: -5 }} />
                            <YAxis stroke="#7e95a6" />
                            <Tooltip contentStyle={{ background: '#0b1823', border: '1px solid #294354' }} />
                            <Legend />
                            <Line type="monotone" dataKey="train_loss" stroke="#4dddc5" name="Train Loss" strokeWidth={2} dot={{ r: 3 }} />
                            <Line type="monotone" dataKey="val_loss" stroke="#ffd166" name="Val Loss" strokeWidth={2} dot={{ r: 3 }} />
                            <Line type="monotone" dataKey="test_loss" stroke="#56e4bd" name="Test Loss" strokeWidth={2} dot={{ r: 3 }} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    ) : (
                      <div className="detailrow">
                        <span>Round Loss</span>
                        <b>{last.test.loss ?? '—'}</b>
                      </div>
                    )}

                    <div className="detailrow">
                      <span>Aggregation Strategy</span>
                      <b>{last.aggregation.toUpperCase()}</b>
                    </div>
                    <div className="detailrow">
                      <span>Data Distribution Partition</span>
                      <b>{last.partition ? last.partition.toUpperCase() : 'IID'}</b>
                    </div>
                    <div className="detailrow">
                      <span>Simulated Attack Vector</span>
                      <b>{last.attack.toUpperCase()}</b>
                    </div>
                    <div className="detailrow">
                      <span>Quarantined Byzantine Clients</span>
                      <b style={{ color: last.flagged_clients.length ? '#ff6974' : '#65dfbd' }}>
                        {last.flagged_clients.length
                          ? last.flagged_clients.map((id) => `Hospital ${id}`).join(', ')
                          : 'None (All honest updates accepted)'}
                      </b>
                    </div>
                    <div className="detailrow">
                      <span>Accepted Model Updates</span>
                      <b>{last.accepted_clients.length} / 10 Hospital Clients</b>
                    </div>
                    <div className="detailrow">
                      <span>Attacker Successfully Isolated</span>
                      <b>
                        {last.detected_attacker === null
                          ? 'N/A (Honest round)'
                          : last.detected_attacker
                          ? 'YES (CORRECTLY ISOLATED BY MAD Z-SCORE)'
                          : 'NO (FALSE NEGATIVE)'}
                      </b>
                    </div>
                    <p className="footnote" style={{ marginTop: 12 }}>
                      Evaluated with PyTorch CNN on independent PneumoniaMNIST held-out split.
                      Model architecture uses two convolutional blocks, batch normalization,
                      ReLU, dropout, and Adam optimization.
                    </p>
                  </>
                ) : (
                  <div className="empty">Run an experiment to inspect aggregation details.</div>
                )}
              </div>
            </section>
          </>
        )}

        {/* TAB 5: AUDIT LOG */}
        {tab === 'AUDIT LOG' && (
          <section className="panel">
            <PanelTitle
              title="SECURITY AUDIT JOURNAL"
              tag={firestoreConnected ? 'FIRESTORE PERSISTED' : 'SQLITE FALLBACK'}
            />
            <div style={{ marginBottom: 12, display: 'flex', gap: 10 }}>
              <button
                className="secondary"
                style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 10 }}
                onClick={() => {
                  if (!currentUser) handleGoogleLogin();
                  else setSheetConfirmOpen(true);
                }}
              >
                <FileSpreadsheet size={13} /> EXPORT AUDIT LOG TO GOOGLE SHEETS
              </button>
            </div>
            {data?.events && data.events.length > 0 ? (
              data.events.map((e, i) => (
                <div className="event" key={i}>
                  <span>{e.time}</span>
                  <b>{e.event}</b>
                  <code>
                    {e.operator && <span style={{ color: '#4dddc5' }}>[{e.operator}] </span>}
                    {JSON.stringify(e.details)}
                  </code>
                </div>
              ))
            ) : (
              <div className="empty">No security events recorded in journal.</div>
            )}
          </section>
        )}

        {/* BOTTOM ROW: DATASET & STATUS */}
        <section className="bottomrow">
          <div className="dataset">
            <Database size={18} />
            {backendStatus === 'ONLINE' && data?.dataset ? (
              <div>
                <b>{data.dataset.name}</b>
                <small>
                  {data.dataset.train_samples} train ·{' '}
                  {data.dataset.val_samples} val ·{' '}
                  {data.dataset.test_samples} test samples{' '}
                  {data.dataset.synthetic ? '· BUILT-IN BENCHMARK' : '· REAL BENCHMARK'}
                </small>
              </div>
            ) : (
              <div>
                <b style={{ color: backendStatus === 'ONLINE' ? '#bad2dc' : '#ff818b' }}>
                  {backendStatus === 'ONLINE' ? 'NO DATASET LOADED' : 'FASTAPI BACKEND OFFLINE'}
                </b>
                <small style={{ color: '#889ea8' }}>
                  {backendStatus === 'ONLINE'
                    ? 'Upload PneumoniaMNIST .npz archive to initialize training'
                    : 'Awaiting telemetry from Python ML backend · No dataset loaded'}
                </small>
              </div>
            )}

            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <label className="upload">
                <Upload size={15} /> {uploadStatus.uploading ? 'UPLOADING…' : 'LOCAL NPZ'}
                <input type="file" accept=".npz" onChange={upload} disabled={uploadStatus.uploading} />
              </label>
              <button
                className="upload"
                style={{ background: 'transparent', cursor: 'pointer' }}
                onClick={handleOpenDrivePicker}
                title="Pick .npz dataset from Google Drive"
                disabled={uploadStatus.uploading}
              >
                <FolderOpen size={15} /> GOOGLE DRIVE
              </button>
            </div>
          </div>
          <div className="footerstatus">
            <span className="pulse" /> {msg}
          </div>
        </section>

        {uploadStatus.error && (
          <div
            style={{
              marginTop: 10,
              padding: '8px 12px',
              background: '#2b1419',
              border: '1px solid #732832',
              color: '#ff818b',
              fontSize: 11,
              fontFamily: 'IBM Plex Mono',
            }}
          >
            Dataset Error: {uploadStatus.error}
          </div>
        )}

        {uploadStatus.summary && (
          <div
            style={{
              marginTop: 10,
              padding: '8px 12px',
              background: '#0d2d2c',
              border: '1px solid #23655d',
              color: '#65dfbd',
              fontSize: 11,
              fontFamily: 'IBM Plex Mono',
            }}
          >
            ✓ Dataset Validated: {uploadStatus.summary.name} — {uploadStatus.summary.train_samples} train / {uploadStatus.summary.val_samples} val / {uploadStatus.summary.test_samples} test ({uploadStatus.summary.features} features).
          </div>
        )}

        <footer>
          COVENGUARD AI <span>·</span> RESEARCH PROTOTYPE <span>·</span> NOT FOR CLINICAL
          DECISION-MAKING
        </footer>
      </main>
    </div>
  );
}

function Metric({
  label,
  value,
  suffix = '',
  icon,
  danger = false,
}: {
  label: string;
  value: string | number;
  suffix?: string;
  icon: React.ReactNode;
  danger?: boolean;
}) {
  return (
    <div className="metric">
      <div className="metric-head">
        <span>{label}</span>
        <i>{icon}</i>
      </div>
      <div className={'metric-value ' + (danger ? 'red' : '')}>
        {value}
        <small>{suffix}</small>
      </div>
    </div>
  );
}

function PanelTitle({ title, tag }: { title: string; tag: string }) {
  return (
    <div className="paneltitle">
      <b>{title}</b>
      <span>{tag}</span>
    </div>
  );
}
