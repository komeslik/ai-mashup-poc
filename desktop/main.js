/**
 * Electron main process — spawn FastAPI sidecar, open UI, quit cleanly.
 *
 * Dev (from repo, any OS):
 *   cd desktop && npm install && npm run dev
 *   → spawns `python ../desktop_server.py` (uses your venv PATH if activated)
 *
 * Packaged Windows:
 *   resources/sidecar/mashup-server.exe (+ ffmpeg.exe)
 */

const { app, BrowserWindow, dialog, shell } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const net = require("net");
const path = require("path");
const fs = require("fs");

const isDev = process.argv.includes("--dev") || !app.isPackaged;

let mainWindow = null;
let serverProcess = null;
let serverPort = 8765;

function findFreePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.listen(0, "127.0.0.1", () => {
      const addr = srv.address();
      const port = typeof addr === "object" && addr ? addr.port : 8765;
      srv.close(() => resolve(port));
    });
    srv.on("error", reject);
  });
}

function waitForHealth(port, timeoutMs = 120000) {
  const started = Date.now();
  return new Promise((resolve, reject) => {
    const tryOnce = () => {
      const req = http.get(
        { host: "127.0.0.1", port, path: "/health", timeout: 2000 },
        (res) => {
          let body = "";
          res.on("data", (c) => {
            body += c;
          });
          res.on("end", () => {
            if (res.statusCode === 200) {
              resolve(body);
              return;
            }
            retry();
          });
        }
      );
      req.on("error", retry);
      req.on("timeout", () => {
        req.destroy();
        retry();
      });
    };
    const retry = () => {
      if (Date.now() - started > timeoutMs) {
        reject(new Error(`Server did not become healthy on port ${port}`));
        return;
      }
      setTimeout(tryOnce, 500);
    };
    tryOnce();
  });
}

function sidecarPaths() {
  if (isDev) {
    const repoRoot = path.resolve(__dirname, "..");
    return {
      mode: "python",
      repoRoot,
      script: path.join(repoRoot, "desktop_server.py"),
      cwd: repoRoot,
    };
  }
  const res = process.resourcesPath;
  const sidecarDir = path.join(res, "sidecar");
  const exeName =
    process.platform === "win32" ? "mashup-server.exe" : "mashup-server";
  return {
    mode: "frozen",
    sidecarDir,
    exe: path.join(sidecarDir, exeName),
    cwd: sidecarDir,
  };
}

function startServer(port) {
  const info = sidecarPaths();
  const env = {
    ...process.env,
    MASHUP_DESKTOP: "1",
    MASHUP_PORT: String(port),
  };

  let child;
  if (info.mode === "python") {
    const py =
      process.env.MASHUP_PYTHON ||
      (process.platform === "win32" ? "python" : "python3");
    child = spawn(py, [info.script, "--host", "127.0.0.1", "--port", String(port)], {
      cwd: info.cwd,
      env,
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    });
  } else {
    if (!fs.existsSync(info.exe)) {
      throw new Error(
        `Sidecar missing at ${info.exe}. Build with scripts/build_sidecar_win.ps1 (CI).`
      );
    }
    // Prefer bundled ffmpeg on PATH for the sidecar process.
    env.PATH = `${info.sidecarDir}${path.delimiter}${env.PATH || ""}`;
    child = spawn(info.exe, ["--host", "127.0.0.1", "--port", String(port)], {
      cwd: info.cwd,
      env,
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    });
  }

  child.stdout.on("data", (d) => console.log(`[sidecar] ${d}`));
  child.stderr.on("data", (d) => console.error(`[sidecar] ${d}`));
  child.on("exit", (code, signal) => {
    console.log(`sidecar exited code=${code} signal=${signal}`);
  });
  return child;
}

function stopServer() {
  if (!serverProcess) return;
  const child = serverProcess;
  serverProcess = null;
  try {
    if (process.platform === "win32") {
      spawn("taskkill", ["/pid", String(child.pid), "/f", "/t"], {
        windowsHide: true,
        stdio: "ignore",
      });
    } else {
      child.kill("SIGTERM");
    }
  } catch (err) {
    console.error("Failed to stop sidecar", err);
  }
}

async function createWindow() {
  serverPort = await findFreePort();
  serverProcess = startServer(serverPort);
  await waitForHealth(serverPort);

  mainWindow = new BrowserWindow({
    width: 1280,
    height: 900,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
    title: "AI Song Mashup",
  });

  mainWindow.loadURL(`http://127.0.0.1:${serverPort}/`);
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });
}

app.whenReady().then(async () => {
  try {
    await createWindow();
  } catch (err) {
    console.error(err);
    dialog.showErrorBox(
      "AI Song Mashup failed to start",
      String(err && err.message ? err.message : err)
    );
    stopServer();
    app.quit();
  }
});

app.on("window-all-closed", () => {
  stopServer();
  app.quit();
});

app.on("before-quit", () => {
  stopServer();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow().catch(console.error);
  }
});
