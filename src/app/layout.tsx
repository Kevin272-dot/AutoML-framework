// src/app/layout.tsx
// Global styles are inlined here as a <style> tag instead of a
// separate globals.css file, to keep the project file count down.
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "IDP AutoML — Dataset Discovery + AutoML",
  description: "Search for a dataset by requirement, then automatically train and compare ML models.",
};

const GLOBAL_CSS = `
  :root { --bg:#0a0e17; --bg-alt:#111827; --card:#161d2b; --text:#e6e9f0; --muted:#9aa4b8;
    --accent:#6d5efc; --accent2:#00d4ff; --border:#232b3d; --success:#22c55e; --danger:#ef4444; --radius:12px; }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,-apple-system,sans-serif; line-height:1.6; }
  a { color:inherit; text-decoration:none; }
  .container { max-width:1100px; margin:0 auto; padding:32px 24px; }
  .navbar { position:sticky; top:0; z-index:50; background:rgba(10,14,23,.9); backdrop-filter:blur(10px); border-bottom:1px solid var(--border); }
  .navbar-inner { max-width:1100px; margin:0 auto; padding:16px 24px; display:flex; align-items:center; justify-content:space-between; }
  .logo { font-size:1.3rem; font-weight:800; }
  .logo span { color:var(--accent2); }
  .steps { display:flex; gap:6px; font-size:.8rem; color:var(--muted); flex-wrap:wrap; }
  .step { padding:4px 10px; border-radius:20px; border:1px solid var(--border); }
  .step.active { color:#fff; border-color:var(--accent); background:rgba(109,94,252,.15); }
  h1 { font-size:2.2rem; font-weight:800; margin-bottom:12px; }
  h2 { font-size:1.4rem; font-weight:700; margin-bottom:16px; }
  p.subtitle { color:var(--muted); margin-bottom:32px; max-width:600px; }
  .search-box { display:flex; gap:12px; margin-bottom:40px; flex-wrap:wrap; }
  input, select, textarea { background:var(--card); border:1px solid var(--border); color:var(--text);
    padding:12px 16px; border-radius:8px; font-size:1rem; flex:1; font-family:inherit; }
  input:focus, select:focus, textarea:focus { outline:none; border-color:var(--accent); }
  .btn { display:inline-flex; align-items:center; gap:8px; padding:12px 24px; border-radius:8px; font-weight:600;
    font-size:.95rem; border:none; cursor:pointer; transition:transform .15s, box-shadow .15s; }
  .btn-primary { background:linear-gradient(135deg,var(--accent),var(--accent2)); color:#fff; }
  .btn-primary:hover { transform:translateY(-2px); box-shadow:0 8px 20px rgba(109,94,252,.35); }
  .btn-primary:disabled { opacity:.5; cursor:not-allowed; transform:none; box-shadow:none; }
  .btn-outline { background:transparent; border:1px solid var(--border); color:var(--text); }
  .btn-outline:hover { border-color:var(--accent); }
  .card { background:var(--card); border:1px solid var(--border); border-radius:var(--radius); padding:24px; margin-bottom:16px; transition:border-color .15s; }
  .card:hover { border-color:var(--accent); }
  .card-title { font-size:1.1rem; font-weight:700; margin-bottom:6px; }
  .card-desc { color:var(--muted); font-size:.9rem; margin-bottom:14px; }
  .badge { display:inline-block; padding:4px 12px; border-radius:16px; font-size:.78rem; font-weight:700; background:var(--bg-alt); border:1px solid var(--border); }
  .badge-score { color:var(--accent2); }
  .badge-domain { color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
  .meta-row { display:flex; gap:18px; flex-wrap:wrap; color:var(--muted); font-size:.85rem; margin-top:10px; }
  table { width:100%; border-collapse:collapse; margin:16px 0; font-size:.9rem; }
  th, td { text-align:left; padding:10px 12px; border-bottom:1px solid var(--border); }
  th { color:var(--muted); font-weight:600; }
  tr.best-row { background:rgba(0,212,255,.06); }
  tr.best-row td:first-child { color:var(--accent2); font-weight:700; }
  .progress-track { background:var(--bg-alt); border-radius:8px; height:10px; overflow:hidden; margin:8px 0 20px; }
  .progress-fill { height:100%; background:linear-gradient(90deg,var(--accent),var(--accent2)); transition:width .4s; }
  .status-pill { padding:4px 12px; border-radius:16px; font-size:.8rem; font-weight:700; }
  .status-queued { background:rgba(154,164,184,.15); color:var(--muted); }
  .status-running { background:rgba(0,212,255,.15); color:var(--accent2); }
  .status-completed { background:rgba(34,197,94,.15); color:var(--success); }
  .status-failed { background:rgba(239,68,68,.15); color:var(--danger); }
  .grid-2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:16px; }
  .error-box { background:rgba(239,68,68,.1); border:1px solid var(--danger); border-radius:8px; padding:16px; color:#fca5a5; margin-bottom:16px; }
  .feature-bar-row { display:flex; align-items:center; gap:10px; margin-bottom:8px; font-size:.85rem; }
  .feature-bar-track { flex:1; background:var(--bg-alt); border-radius:4px; height:8px; overflow:hidden; }
  .feature-bar-fill { height:100%; background:linear-gradient(90deg,var(--accent),var(--accent2)); }
  .feature-label { width:160px; color:var(--muted); }
  .spinner { width:18px; height:18px; border:2px solid var(--border); border-top-color:var(--accent2); border-radius:50%; animation:spin .8s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <style>{GLOBAL_CSS}</style>
      </head>
      <body>
        <nav className="navbar">
          <div className="navbar-inner">
            <a href="/" className="logo">IDP<span>AutoML</span></a>
          </div>
        </nav>
        {children}
      </body>
    </html>
  );
}
