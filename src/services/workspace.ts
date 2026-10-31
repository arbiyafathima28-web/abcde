import {
  GoogleAuthProvider,
  signInWithPopup,
  signOut,
  onAuthStateChanged,
  User,
} from 'firebase/auth';
import { auth } from './firebase';

export const WORKSPACE_SCOPES = [
  'https://www.googleapis.com/auth/drive.file',
  'https://www.googleapis.com/auth/drive.metadata.readonly',
  'https://www.googleapis.com/auth/drive',
  'https://www.googleapis.com/auth/drive.readonly',
  'https://www.googleapis.com/auth/spreadsheets',
  'https://www.googleapis.com/auth/spreadsheets.readonly',
];

const provider = new GoogleAuthProvider();
WORKSPACE_SCOPES.forEach((scope) => provider.addScope(scope));

let isSigningIn = false;
let cachedAccessToken: string | null = null;

export const initAuth = (
  onAuthSuccess?: (user: User, token: string) => void,
  onAuthFailure?: () => void
) => {
  return onAuthStateChanged(auth, async (user: User | null) => {
    if (user) {
      if (cachedAccessToken) {
        if (onAuthSuccess) onAuthSuccess(user, cachedAccessToken);
      } else if (!isSigningIn) {
        // Token must be acquired through explicit user sign-in action in popup flow
        if (onAuthFailure) onAuthFailure();
      }
    } else {
      cachedAccessToken = null;
      if (onAuthFailure) onAuthFailure();
    }
  });
};

export const googleSignIn = async (): Promise<{ user: User; accessToken: string } | null> => {
  try {
    isSigningIn = true;
    const result = await signInWithPopup(auth, provider);
    const credential = GoogleAuthProvider.credentialFromResult(result);
    if (!credential?.accessToken) {
      throw new Error('Failed to acquire OAuth access token from Google sign-in');
    }
    cachedAccessToken = credential.accessToken;
    return { user: result.user, accessToken: cachedAccessToken };
  } catch (error: any) {
    console.error('Google Sign In error:', error);
    throw error;
  } finally {
    isSigningIn = false;
  }
};

export const getAccessToken = async (): Promise<string | null> => {
  return cachedAccessToken;
};

export const logout = async () => {
  await signOut(auth);
  cachedAccessToken = null;
};

// --- Google Picker API Integration ---

declare global {
  interface Window {
    gapi: any;
    google: any;
  }
}

let pickerApiLoaded = false;

function loadGapiScript(): Promise<void> {
  return new Promise((resolve, reject) => {
    if (window.gapi) {
      resolve();
      return;
    }
    const script = document.createElement('script');
    script.src = 'https://apis.google.com/js/api.js';
    script.async = true;
    script.defer = true;
    script.onload = () => resolve();
    script.onerror = (e) => reject(new Error('Failed to load Google API script'));
    document.body.appendChild(script);
  });
}

export async function openGooglePicker(
  onFilePicked: (file: { id: string; name: string; mimeType: string }) => void
): Promise<void> {
  const token = await getAccessToken();
  if (!token) {
    throw new Error('Please sign in with Google first to access Google Drive.');
  }

  await loadGapiScript();

  if (!pickerApiLoaded) {
    await new Promise<void>((resolve) => {
      window.gapi.load('picker', () => {
        pickerApiLoaded = true;
        resolve();
      });
    });
  }

  const pickerOrigin =
    window.location.ancestorOrigins && window.location.ancestorOrigins.length > 0
      ? window.location.ancestorOrigins[window.location.ancestorOrigins.length - 1]
      : window.location.origin;

  const docsView = new window.google.picker.DocsView()
    .setIncludeFolders(true)
    .setSelectFolderEnabled(false);

  const picker = new window.google.picker.PickerBuilder()
    .addView(docsView)
    .addView(new window.google.picker.DocsUploadView())
    .setOAuthToken(token)
    .setOrigin(pickerOrigin)
    .setCallback((data: any) => {
      if (data.action === window.google.picker.Action.PICKED) {
        const file = data.docs[0];
        onFilePicked({
          id: file.id,
          name: file.name,
          mimeType: file.mimeType,
        });
      }
    })
    .build();

  picker.setVisible(true);
}

// Download file bytes from Google Drive using Bearer token
export async function downloadDriveFile(fileId: string): Promise<Blob> {
  const token = await getAccessToken();
  if (!token) throw new Error('Not authenticated with Google Workspace');

  const res = await fetch(`https://www.googleapis.com/drive/v3/files/${fileId}?alt=media`, {
    headers: { Authorization: `Bearer ${token}` },
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Drive download failed (HTTP ${res.status}): ${err}`);
  }

  return await res.blob();
}

// --- Google Sheets API Integration ---

export interface SheetExportPayload {
  round: number;
  attack: string;
  aggregation: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  clients: {
    id: number;
    name: string;
    status: string;
    trust: number;
    samples: number;
    l2_norm?: number;
    mean_cosine?: number;
    robust_score?: number;
  }[];
  events: { time: string; event: string; details: any }[];
}

export async function exportToGoogleSheets(
  payload: SheetExportPayload
): Promise<{ spreadsheetId: string; spreadsheetUrl: string }> {
  const token = await getAccessToken();
  if (!token) throw new Error('Please sign in with Google to export to Google Sheets.');

  const title = `CovenGuard AI - Round ${payload.round} Telemetry Audit (${new Date().toLocaleDateString()})`;

  // Step 1: Create a new spreadsheet with customized title and tabs
  const createRes = await fetch('https://sheets.googleapis.com/v4/spreadsheets', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      properties: { title },
      sheets: [
        { properties: { title: 'Executive Summary' } },
        { properties: { title: 'Hospital Trust Ledger' } },
        { properties: { title: 'Security Audit Journal' } },
      ],
    }),
  });

  if (!createRes.ok) {
    const err = await createRes.text();
    throw new Error(`Failed to create Google Spreadsheet: ${err}`);
  }

  const sheetData = await createRes.json();
  const spreadsheetId = sheetData.spreadsheetId;
  const spreadsheetUrl =
    sheetData.spreadsheetUrl || `https://docs.google.com/spreadsheets/d/${spreadsheetId}/edit`;

  // Step 2: Populate Data via batchUpdate
  const summaryValues = [
    ['COVENGUARD AI - FEDERATED DEFENCE AUDIT REPORT', ''],
    ['Generated On', new Date().toISOString()],
    ['Federated Round', payload.round],
    ['Simulated Attack Vector', payload.attack.toUpperCase()],
    ['Consensus Aggregation Algorithm', payload.aggregation.toUpperCase()],
    ['Global Test Accuracy', `${(payload.accuracy * 100).toFixed(2)}%`],
    ['Global Precision', `${(payload.precision * 100).toFixed(2)}%`],
    ['Global Recall', `${(payload.recall * 100).toFixed(2)}%`],
    ['Global F1 Score', `${(payload.f1 * 100).toFixed(2)}%`],
  ];

  const ledgerHeaders = [
    'Node ID',
    'Facility Name',
    'Security Status',
    'Trust Score (%)',
    'Partition Samples',
    'L2 Update Norm',
    'Mean Cosine Alignment',
    'Robust MAD z-Score',
  ];
  const ledgerRows = payload.clients.map((c) => [
    `H-${String(c.id).padStart(2, '0')}`,
    c.name,
    c.status,
    c.trust,
    c.samples,
    c.l2_norm ?? 'N/A',
    c.mean_cosine ?? 'N/A',
    c.robust_score ?? 'N/A',
  ]);
  const ledgerValues = [ledgerHeaders, ...ledgerRows];

  const auditHeaders = ['Timestamp', 'Security Event', 'Telemetry Details'];
  const auditRows = payload.events.map((e) => [
    e.time,
    e.event,
    typeof e.details === 'string' ? e.details : JSON.stringify(e.details),
  ]);
  const auditValues = [auditHeaders, ...auditRows];

  const updateRes = await fetch(
    `https://sheets.googleapis.com/v4/spreadsheets/${spreadsheetId}/values:batchUpdate`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        valueInputOption: 'USER_ENTERED',
        data: [
          {
            range: "'Executive Summary'!A1",
            values: summaryValues,
          },
          {
            range: "'Hospital Trust Ledger'!A1",
            values: ledgerValues,
          },
          {
            range: "'Security Audit Journal'!A1",
            values: auditValues,
          },
        ],
      }),
    }
  );

  if (!updateRes.ok) {
    const err = await updateRes.text();
    throw new Error(`Failed to write telemetry data to Google Sheet: ${err}`);
  }

  return { spreadsheetId, spreadsheetUrl };
}
