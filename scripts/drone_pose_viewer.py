#!/usr/bin/env python3
"""Stream the drone's live pose over a WebSocket and draw it in a browser.

    uv run scripts/drone_pose_viewer.py
    then open http://127.0.0.1:8000

Pose is in the inertial frame the ArUco calibration fixed: origin at the marker
centre, +Z up. The cameras are read on the main thread; the HTTP/WebSocket
server runs alongside and pushes whatever the latest pose is, so a slow or
absent browser never stalls the tracking loop.

Only the server -> client half of the WebSocket protocol is implemented, which
is all a one-way telemetry feed needs, and it keeps the project dependency-free.
"""
import argparse
import base64
import hashlib
import json
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from drone.pose import PoseTracker

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

_latest = {"t": 0.0, "blobs": [0, 0]}
_lock = threading.Lock()


def ws_frame(payload):
    """Wrap bytes in a single unmasked text frame (FIN + opcode 1)."""
    n = len(payload)
    if n < 126:
        head = struct.pack("!BB", 0x81, n)
    elif n < 1 << 16:
        head = struct.pack("!BBH", 0x81, 126, n)
    else:
        head = struct.pack("!BBQ", 0x81, 127, n)
    return head + payload


PAGE = """<!doctype html>
<meta charset="utf-8">
<title>drone pose</title>
<style>
  body { margin:0; background:#111; color:#ddd;
         font:12px ui-monospace,Menlo,Consolas,monospace; }
  h2 { font-size:12px; font-weight:normal; letter-spacing:.12em;
       text-transform:uppercase; color:#888; margin:18px 16px 6px; }
  .row { display:flex; flex-wrap:wrap; gap:14px; padding:0 16px; }
  .cell div { color:#777; margin-bottom:4px; }
  canvas { background:#181818; border:1px solid #333; display:block;
           touch-action:none; }
  #read { padding:16px; white-space:pre; line-height:1.65; }
  .warn { color:#e55; }
</style>

<h2>raw pose &nbsp;<span style="color:#fa0">&#9679;</span></h2>
<div class="row">
  <div class="cell"><div>top &nbsp; X right, Y up</div><canvas id="rawtop" width="300" height="300"></canvas></div>
  <div class="cell"><div>side &nbsp; X right, Z up</div><canvas id="rawside" width="300" height="300"></canvas></div>
  <div class="cell"><div>3D &nbsp; drag to orbit</div><canvas id="raw3d" width="300" height="300"></canvas></div>
</div>

<h2>EKF estimate &nbsp;<span style="color:#3cf">&#9679;</span></h2>
<div class="row">
  <div class="cell"><div>top &nbsp; X right, Y up</div><canvas id="esttop" width="300" height="300"></canvas></div>
  <div class="cell"><div>side &nbsp; X right, Z up</div><canvas id="estside" width="300" height="300"></canvas></div>
  <div class="cell"><div>3D &nbsp; drag to orbit</div><canvas id="est3d" width="300" height="300"></canvas></div>
</div>

<div id="read">connecting...</div>
<script>
const EXTENT = 1.5;                      // metres shown from the origin
let msg = null, lastRx = 0;
let az = -0.9, el = 0.9;                 // shared orbit for both 3D views

function rotFromRpy(rpy) {               // ZYX, degrees -> body->world matrix
  const [r, p, y] = rpy.map(d => d * Math.PI / 180);
  const cr=Math.cos(r), sr=Math.sin(r), cp=Math.cos(p), sp=Math.sin(p),
        cy=Math.cos(y), sy=Math.sin(y);
  return [[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
          [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
          [ -sp,            cp*sr,            cp*cr]];
}
const add = (a,b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
const mul = (R,v) => [R[0][0]*v[0]+R[0][1]*v[1]+R[0][2]*v[2],
                      R[1][0]*v[0]+R[1][1]*v[1]+R[1][2]*v[2],
                      R[2][0]*v[0]+R[2][1]*v[1]+R[2][2]*v[2]];

function frame(cv) {
  const g = cv.getContext("2d");
  g.clearRect(0, 0, cv.width, cv.height);
  return g;
}

// ---- flat views ---------------------------------------------------------
function draw2D(cv, po, hAxis, vAxis, floor, col) {
  const g = frame(cv), W = cv.width, H = cv.height, s = (W/2)/EXTENT;
  const px = v => W/2 + v*s, py = v => H/2 - v*s;

  g.strokeStyle = "#282828"; g.lineWidth = 1;
  for (let m = -EXTENT; m <= EXTENT+1e-9; m += 0.5) {
    g.beginPath(); g.moveTo(px(m), 0); g.lineTo(px(m), H); g.stroke();
    g.beginPath(); g.moveTo(0, py(m)); g.lineTo(W, py(m)); g.stroke();
  }
  if (floor) { g.strokeStyle="#444"; g.lineWidth=2;
    g.beginPath(); g.moveTo(0, py(0)); g.lineTo(W, py(0)); g.stroke(); }
  g.lineWidth = 1;
  g.strokeStyle="#c33"; g.beginPath(); g.moveTo(px(0),py(0)); g.lineTo(px(0.3),py(0)); g.stroke();
  g.strokeStyle="#3c3"; g.beginPath(); g.moveTo(px(0),py(0)); g.lineTo(px(0),py(0.3)); g.stroke();

  if (msg && msg.cams) {                 // where the cameras sit
    g.fillStyle = "#456";
    for (const C of msg.cams) {
      g.fillRect(px(C[hAxis])-3, py(C[vAxis])-3, 6, 6);
    }
  }
  if (!po || !po.pos) return;
  const h = po.pos[hAxis], v = po.pos[vAxis];

  if (po.leds) {
    g.strokeStyle = "#556"; g.beginPath();
    po.leds.forEach((L,i) => i ? g.lineTo(px(L[hAxis]),py(L[vAxis]))
                               : g.moveTo(px(L[hAxis]),py(L[vAxis])));
    g.closePath(); g.stroke();
    g.fillStyle = "#88f";
    for (const L of po.leds) { g.beginPath(); g.arc(px(L[hAxis]),py(L[vAxis]),2.5,0,7); g.fill(); }
  }
  g.fillStyle = col; g.beginPath(); g.arc(px(h), py(v), 5, 0, 7); g.fill();

  const R = rotFromRpy(po.rpy), fwd = add(po.pos, mul(R,[0.3,0,0]));
  g.strokeStyle = col; g.lineWidth = 2; g.beginPath();
  g.moveTo(px(h), py(v)); g.lineTo(px(fwd[hAxis]), py(fwd[vAxis])); g.stroke();
}

// ---- 3D view ------------------------------------------------------------
function draw3D(cv, po, col) {
  const g = frame(cv), W = cv.width, H = cv.height, s = (W/2)/(EXTENT*1.45);
  const ca=Math.cos(az), sa=Math.sin(az), ce=Math.cos(el), se=Math.sin(el);
  const proj = P => {                    // orthographic: yaw by az, tilt by el
    const X  =  P[0]*ca + P[1]*sa;
    const Yp = -P[0]*sa + P[1]*ca;
    return [W/2 + X*s, H/2 + 0.35*H/2 - (P[2]*ce - Yp*se)*s];
  };
  const line = (a, b) => { const A=proj(a), B=proj(b);
    g.beginPath(); g.moveTo(A[0],A[1]); g.lineTo(B[0],B[1]); g.stroke(); };

  g.strokeStyle = "#282828"; g.lineWidth = 1;          // floor grid, z = 0
  for (let m = -EXTENT; m <= EXTENT+1e-9; m += 0.5) {
    line([m,-EXTENT,0], [m,EXTENT,0]);
    line([-EXTENT,m,0], [EXTENT,m,0]);
  }
  g.lineWidth = 2;                                     // world axes at origin
  g.strokeStyle="#c33"; line([0,0,0],[0.4,0,0]);
  g.strokeStyle="#3c3"; line([0,0,0],[0,0.4,0]);
  g.strokeStyle="#46f"; line([0,0,0],[0,0,0.4]);
  g.lineWidth = 1;

  if (msg && msg.cams) {
    g.fillStyle = "#456";
    for (const C of msg.cams) { const A = proj(C);
      g.fillRect(A[0]-3, A[1]-3, 6, 6);
      g.strokeStyle="#333"; line(C, [C[0],C[1],0]); }
  }
  if (!po || !po.pos) return;

  g.strokeStyle="#333"; line(po.pos, [po.pos[0],po.pos[1],0]);   // drop line

  if (po.leds) {
    g.strokeStyle = "#556"; g.beginPath();
    po.leds.forEach((L,i) => { const A=proj(L); i ? g.lineTo(A[0],A[1]) : g.moveTo(A[0],A[1]); });
    g.closePath(); g.stroke();
    g.fillStyle = "#88f";
    for (const L of po.leds) { const A=proj(L);
      g.beginPath(); g.arc(A[0],A[1],2.5,0,7); g.fill(); }
  }
  const R = rotFromRpy(po.rpy);                        // body axes
  g.lineWidth = 2;
  const cols = ["#e66","#6e6","#68f"];
  for (let k = 0; k < 3; k++) {
    const e = [0,0,0]; e[k] = 0.25;
    g.strokeStyle = cols[k]; line(po.pos, add(po.pos, mul(R,e)));
  }
  g.lineWidth = 1;
  const A = proj(po.pos);
  g.fillStyle = col; g.beginPath(); g.arc(A[0],A[1],5,0,7); g.fill();
}

// ---- orbit --------------------------------------------------------------
for (const id of ["raw3d","est3d"]) {
  const cv = document.getElementById(id);
  let drag = null;
  cv.addEventListener("pointerdown", e => { drag = [e.clientX, e.clientY];
                                            cv.setPointerCapture(e.pointerId); });
  cv.addEventListener("pointermove", e => {
    if (!drag) return;
    az -= (e.clientX - drag[0]) * 0.01;
    el = Math.max(-1.5, Math.min(1.5, el + (e.clientY - drag[1]) * 0.01));
    drag = [e.clientX, e.clientY];
  });
  cv.addEventListener("pointerup", () => { drag = null; });
}

// ---- render loop --------------------------------------------------------
const f = v => (v >= 0 ? "+" : "") + v.toFixed(3);

function render() {
  const raw = msg && msg.raw, est = msg && msg.est;
  const rawCol = (raw && raw.ok) ? "#fa0" : "#e55";
  const estCol = (est && est.coasting) ? "#fa0" : "#3cf";

  draw2D(document.getElementById("rawtop"),  raw, 0, 1, false, rawCol);
  draw2D(document.getElementById("rawside"), raw, 0, 2, true,  rawCol);
  draw3D(document.getElementById("raw3d"),   raw, rawCol);
  draw2D(document.getElementById("esttop"),  est, 0, 1, false, estCol);
  draw2D(document.getElementById("estside"), est, 0, 2, true,  estCol);
  draw3D(document.getElementById("est3d"),   est, estCol);

  const el_ = document.getElementById("read");
  if (!msg) { el_.textContent = "waiting for data..."; return; }
  const stale = (Date.now() - lastRx) / 1000 > 1;
  el_.className = stale ? "warn" : "";

  let s = `blobs      cam0 ${msg.blobs[0]}/3   cam1 ${msg.blobs[1]}/3` +
          (stale ? "     [feed stale]" : "") + "\\n\\n";
  s += raw && raw.pos
    ? `raw   pos  ${f(raw.pos[0])} ${f(raw.pos[1])} ${f(raw.pos[2])} m` +
      `   rpy ${f(raw.rpy[0])} ${f(raw.rpy[1])} ${f(raw.rpy[2])} deg\\n` +
      `      fit  rmsd ${(raw.rmsd*1000).toFixed(1)} mm   reproj ${raw.reproj.toFixed(2)} px` +
      (raw.ok ? "" : "   <- rejected, LED triangle does not fit") + "\\n"
    : "raw   no pose -- need 3 blobs in both cameras\\n";
  s += "\\n";
  s += est
    ? `EKF   pos  ${f(est.pos[0])} ${f(est.pos[1])} ${f(est.pos[2])} m` +
      `   rpy ${f(est.rpy[0])} ${f(est.rpy[1])} ${f(est.rpy[2])} deg\\n` +
      `      vel  ${f(est.vel[0])} ${f(est.vel[1])} ${f(est.vel[2])} m/s` +
      `   sigma ${(est.pos_std[0]*1000).toFixed(1)}/${(est.pos_std[1]*1000).toFixed(1)}` +
      `/${(est.pos_std[2]*1000).toFixed(1)} mm\\n` +
      `      upd  accepted ${est.accepted}  rejected ${est.rejected}` +
      (est.coasting ? `   <- coasting ${(est.age*1000).toFixed(0)} ms` : "")
    : "EKF   not initialised -- waiting for a trusted raw pose";
  el_.textContent = s;
}

const ws = new WebSocket("ws://" + location.host + "/ws");
ws.onmessage = e => { msg = JSON.parse(e.data); lastRx = Date.now(); };
ws.onclose = () => { document.getElementById("read").className = "warn"; };
setInterval(render, 50);
</script>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass                                    # keep the pose console readable

    def do_GET(self):
        if self.path.rstrip("/") == "/ws":
            self.serve_ws()
        else:
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def serve_ws(self):
        key = self.headers.get("Sec-WebSocket-Key")
        if key is None:
            self.send_error(400, "not a WebSocket handshake")
            return
        accept = base64.b64encode(
            hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
        self.wfile.write(
            b"HTTP/1.1 101 Switching Protocols\r\n"
            b"Upgrade: websocket\r\n"
            b"Connection: Upgrade\r\n"
            b"Sec-WebSocket-Accept: " + accept.encode() + b"\r\n\r\n")
        self.wfile.flush()

        period = 1.0 / self.server.send_hz
        while True:
            with _lock:
                payload = json.dumps(_latest).encode()
            self.wfile.write(ws_frame(payload))
            self.wfile.flush()
            time.sleep(period)

    def handle_one_request(self):
        # a dropped browser tab is normal, not an error worth a traceback
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[1] / "config")
    ap.add_argument("--host", default="127.0.0.1",
                    help="use 0.0.0.0 to view from another machine")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--send-hz", type=float, default=30.0,
                    help="WebSocket push rate")
    args = ap.parse_args()

    tracker = PoseTracker(args.config)
    tracker.open()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.send_hz = args.send_hz
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"viewer on http://{args.host}:{args.port}   (ctrl-c to stop)")

    try:
        while True:
            pose = tracker.read()
            with _lock:
                _latest.clear()
                _latest.update(pose)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.shutdown()
        tracker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
