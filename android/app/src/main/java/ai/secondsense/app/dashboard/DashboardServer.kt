package ai.secondsense.app.dashboard

import android.content.Context
import android.net.wifi.WifiManager
import android.text.format.Formatter
import android.util.Log
import fi.iki.elonen.NanoHTTPD
import java.net.NetworkInterface
import java.util.Collections

/**
 * Ticket #30 — a tiny embedded HTTP server so a laptop on the SAME local network (the
 * phone's own Wi-Fi/hotspot — no internet involved anywhere, required for the airplane-mode
 * demo, #31) can watch a live dashboard mirroring the on-device HUD. MainActivity calls
 * [publish] once per frame with a JSON snapshot; every browser polling [/state.json] gets it.
 *
 * Deliberately just telemetry, not a video stream — the real UI is audio+haptics for the
 * user (Bible), this is a SPECTATOR view for judges/demo-partners, kept lightweight.
 */
class DashboardServer(port: Int = 8085) : NanoHTTPD(port) {

    @Volatile private var stateJson: String = "{}"
    @Volatile private var frameJpeg: ByteArray? = null

    /** Call once per frame with a JSON string describing the current pipeline state. */
    fun publish(json: String) {
        stateJson = json
    }

    /**
     * Latest camera frame as JPEG bytes, for the laptop-side 3D room demo (laptop/room3d/).
     * Optional, debug-only: a spectator on the same Wi-Fi can GET /frame.jpg. No effect on the
     * on-device pipeline beyond one JPEG encode on a throttled subset of frames.
     */
    fun publishFrame(jpeg: ByteArray) {
        frameJpeg = jpeg
    }

    /**
     * [start] but tolerant of the port still being held by a just-killed previous instance of
     * the app (the socket can linger for a second or two after force-stop). Retries a few times
     * before giving up. Returns true if the server is listening. Call OFF the main thread.
     */
    fun startWithRetry(attempts: Int = 6, delayMs: Long = 400L): Boolean {
        repeat(attempts) { i ->
            try {
                start(SOCKET_READ_TIMEOUT, false)
                Log.i(TAG, "listening on :$listeningPort (attempt ${i + 1})")
                return true
            } catch (e: java.io.IOException) {
                Log.w(TAG, "bind attempt ${i + 1}/$attempts failed: ${e.message}")
                try {
                    Thread.sleep(delayMs)
                } catch (_: InterruptedException) {
                    return false
                }
            }
        }
        return false
    }

    override fun serve(session: IHTTPSession): Response {
        return when (session.uri) {
            "/state.json" -> newFixedLengthResponse(Response.Status.OK, "application/json", stateJson)
            "/frame.jpg" -> {
                val f = frameJpeg
                if (f == null) newFixedLengthResponse(Response.Status.NO_CONTENT, "text/plain", "")
                else newFixedLengthResponse(Response.Status.OK, "image/jpeg", f.inputStream(), f.size.toLong())
            }
            else -> newFixedLengthResponse(Response.Status.OK, "text/html", DASHBOARD_HTML)
        }
    }

    companion object {
        private const val TAG = "SecondSense/dashboard"

        /**
         * First non-loopback IPv4 address. Primary path is NetworkInterface (works whether
         * the phone is a Wi-Fi CLIENT or acting as its own hotspot — the realistic
         * airplane-mode demo setup, #31 — unlike WifiManager, which only reports the address
         * in client mode). Some ROMs restrict interface visibility for regular apps though
         * (observed on this MIUI test phone: NetworkInterface enumeration silently returned
         * nothing despite `adb shell ip addr` showing a real address), so [context] enables a
         * WifiManager fallback for client-mode Wi-Fi specifically.
         */
        fun localIpAddress(context: Context? = null): String? {
            val fromInterfaces = try {
                Collections.list(NetworkInterface.getNetworkInterfaces())
                    .asSequence()
                    .flatMap { Collections.list(it.inetAddresses).asSequence() }
                    .filterNot { it.isLoopbackAddress }
                    .map { it.hostAddress }
                    .firstOrNull { it != null && !it.contains(':') } // IPv4 only, skip link-local IPv6
            } catch (t: Throwable) {
                Log.w(TAG, "NetworkInterface lookup failed: ${t.message}")
                null
            }
            if (fromInterfaces != null) {
                Log.i(TAG, "localIpAddress via NetworkInterface: $fromInterfaces")
                return fromInterfaces
            }
            if (context == null) {
                Log.w(TAG, "localIpAddress: NetworkInterface found nothing, no context for WifiManager fallback")
                return null
            }
            return try {
                val wm = context.applicationContext.getSystemService(Context.WIFI_SERVICE) as? WifiManager
                val ipInt = wm?.connectionInfo?.ipAddress ?: 0
                val addr = if (ipInt != 0) Formatter.formatIpAddress(ipInt) else null
                Log.i(TAG, "localIpAddress via WifiManager fallback: $addr")
                addr
            } catch (t: Throwable) {
                Log.w(TAG, "WifiManager fallback failed: ${t.message}")
                null
            }
        }

        private val DASHBOARD_HTML = """
            <!doctype html><html><head><meta charset="utf-8">
            <title>SecondSense — Live Dashboard</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
              :root { --tier-color: #33384250; }
              body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#0b0f14; color:#e6edf3; margin:0; padding:24px; transition: background .4s; }
              h1 { font-size: 20px; margin:0 0 4px; }
              .sub { color:#8b98a5; font-size:13px; margin-bottom:16px; }
              /* #32 — the tier overlay: a full-width color bar fixed to the top, unmissable
                 from across a room, plus a matching page-edge glow. This is the "dashboard
                 overlay" ticket — a big-room-visible confidence readout, not just a number
                 buried in a card. */
              #tierBar { position:fixed; top:0; left:0; right:0; height:10px; background:var(--tier-color); transition: background .3s; z-index:10; }
              body { box-shadow: inset 0 0 0 3px var(--tier-color); transition: box-shadow .3s; }
              .tier-WHITE { --tier-color: #ededed; } .tier-BLUE { --tier-color: #3B82F6; } .tier-RED { --tier-color: #EF4444; }
              @keyframes redpulse { 0%,100% { opacity:1; } 50% { opacity:.35; } }
              .tier-RED #tierBar { animation: redpulse 0.8s infinite; }
              .grid { display:grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap:14px; margin-top:8px; }
              .card { background:#151b23; border:1px solid #222933; border-radius:12px; padding:16px; }
              .label { font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:#8b98a5; margin-bottom:6px; }
              .val { font-size:22px; font-weight:600; }
              .tier-txt-WHITE { color:#ededed; } .tier-txt-BLUE { color:#3B82F6; } .tier-txt-RED { color:#EF4444; }
              .warn { color:#EF4444; font-weight:700; }
              .det { padding:6px 0; border-bottom:1px solid #222933; font-size:14px; }
              .det:last-child { border-bottom:none; }
              #err { color:#EF4444; display:none; margin-top:12px; }
              /* #32 — rolling confidence-tier history: at-a-glance stability over the last
                 ~50 frames, not just the instantaneous value (a stable WHITE run reads very
                 differently from a flickering WHITE/RED/BLUE mess even at the same instant). */
              #histWrap { margin-top:14px; }
              #hist { display:flex; gap:2px; height:28px; align-items:flex-end; }
              #hist div { flex:1; min-width:3px; height:100%; border-radius:2px; background:#222933; }
            </style></head>
            <body>
              <div id="tierBar"></div>
              <h1>SecondSense — Live Dashboard</h1>
              <div class="sub">Ticket #30/#32 — mirrors the on-device HUD + confidence tier over the local network. No internet used.</div>
              <div class="grid">
                <div class="card"><div class="label">Engine / Mode</div><div class="val" id="engine">—</div></div>
                <div class="card"><div class="label">Inference</div><div class="val" id="infer">—</div></div>
                <div class="card"><div class="label">Current Cue</div><div class="val" id="cue">—</div></div>
                <div class="card"><div class="label">Confidence Tier</div><div class="val" id="tier">—</div></div>
                <div class="card"><div class="label">Drop-off</div><div class="val" id="dropoff">—</div></div>
                <div class="card"><div class="label">Goal (voice)</div><div class="val" id="goal">—</div></div>
              </div>
              <div class="card" id="histWrap">
                <div class="label">Confidence tier — last ~50 frames</div>
                <div id="hist"></div>
              </div>
              <div class="card" style="margin-top:14px;">
                <div class="label">Detections this frame</div>
                <div id="dets">—</div>
              </div>
              <div id="err">Lost connection to phone — check it's still on the same network.</div>
              <script>
                const tierHistory = [];
                const HIST_LEN = 50;
                const tierBarColor = { WHITE: '#ededed', BLUE: '#3B82F6', RED: '#EF4444' };

                function renderHistory() {
                  const hist = document.getElementById('hist');
                  hist.innerHTML = tierHistory.map(t =>
                    '<div style="background:' + (tierBarColor[t] || '#222933') + '"></div>'
                  ).join('');
                }

                async function tick() {
                  try {
                    const r = await fetch('/state.json', {cache:'no-store'});
                    const s = await r.json();
                    document.getElementById('err').style.display = 'none';
                    document.getElementById('engine').textContent = s.engine + ' · ' + s.mode;
                    document.getElementById('infer').textContent = s.inferMs + ' ms  (frame ' + s.frames + ')';
                    document.getElementById('cue').textContent = s.cueLabel ? (s.cueLabel + ' ' + s.cueDir + ' p=' + s.cueProx) : '—';

                    const tier = s.tier || null;
                    const tierEl = document.getElementById('tier');
                    tierEl.textContent = tier || '—';
                    tierEl.className = 'val tier-txt-' + (tier || '');
                    // #32 overlay: drive the full-page tier bar/glow + RED pulse from the SAME
                    // value, so the big room-visible signal and the text always agree.
                    document.body.className = tier ? ('tier-' + tier) : '';

                    tierHistory.push(tier || 'none');
                    if (tierHistory.length > HIST_LEN) tierHistory.shift();
                    renderHistory();

                    document.getElementById('dropoff').innerHTML = s.dropOff
                      ? ('<span class="warn">⚠ AHEAD' + (s.dropOffPct != null ? (' (' + s.dropOffPct + '% down)') : '') + (s.dropOffBaroConfirmed ? ' — BAROMETER CONFIRMED' : '') + '</span>')
                      : 'clear';
                    document.getElementById('goal').textContent = s.goal || '—';
                    const dets = (s.detections || []);
                    document.getElementById('dets').innerHTML = dets.length
                      ? dets.map(d => '<div class="det">' + (d.label || '(unknown)') + ' — score ' + d.score + ' — prox ' + d.prox + '</div>').join('')
                      : '<div class="det">(none)</div>';
                  } catch (e) {
                    document.getElementById('err').style.display = 'block';
                  }
                  setTimeout(tick, 400);
                }
                tick();
              </script>
            </body></html>
        """.trimIndent()
    }
}
