(() => {
  const zoneA = document.getElementById("zone-a");
  const zoneB = document.getElementById("zone-b");
  const fileA = document.getElementById("file-a");
  const fileB = document.getElementById("file-b");
  const nameA = document.getElementById("name-a");
  const nameB = document.getElementById("name-b");
  const mashupBtn = document.getElementById("mashup-btn");
  const statusPanel = document.getElementById("status-panel");
  const progressWrap = document.getElementById("progress-wrap");
  const progressBar = document.getElementById("progress-bar");
  const progressLabel = document.getElementById("progress-label");
  const elapsedEl = document.getElementById("elapsed");
  const errorMsg = document.getElementById("error-msg");
  const downloadBtn = document.getElementById("download-btn");

  /** @type {File | null} */
  let songA = null;
  /** @type {File | null} */
  let songB = null;
  /** @type {string | null} */
  let objectUrl = null;

  const STAGE_MESSAGES = [
    "Uploading tracks…",
    "Separating stems with local Demucs…",
    "Detecting BPM…",
    "Choosing mix strategy…",
    "Time-stretching vocals…",
    "Matching vocal key to instrumental…",
    "Mixing stems…",
    "Almost there…",
  ];

  function isAudioFile(file) {
    if (!file) return false;
    if (file.type && file.type.startsWith("audio/")) return true;
    return /\.(mp3|wav|flac|m4a|aac|ogg|aiff|aif)$/i.test(file.name);
  }

  function formatElapsed(ms) {
    const totalSec = Math.floor(ms / 1000);
    const m = Math.floor(totalSec / 60);
    const s = totalSec % 60;
    return `${m}:${String(s).padStart(2, "0")}`;
  }

  function updateButton() {
    mashupBtn.disabled = !(songA && songB) || mashupBtn.classList.contains("busy");
  }

  function setFile(slot, file) {
    if (!isAudioFile(file)) {
      showError("Please drop an audio file (mp3, wav, flac, m4a…).");
      return;
    }
    hideError();
    if (slot === "a") {
      songA = file;
      nameA.textContent = file.name;
      zoneA.classList.add("filled");
    } else {
      songB = file;
      nameB.textContent = file.name;
      zoneB.classList.add("filled");
    }
    updateButton();
  }

  function wireZone(zone, input, slot) {
    zone.addEventListener("click", () => input.click());
    zone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        input.click();
      }
    });
    input.addEventListener("change", () => {
      const file = input.files && input.files[0];
      if (file) setFile(slot, file);
    });

    zone.addEventListener("dragenter", (event) => {
      event.preventDefault();
      zone.classList.add("dragover");
    });
    zone.addEventListener("dragover", (event) => {
      event.preventDefault();
      zone.classList.add("dragover");
    });
    zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
    zone.addEventListener("drop", (event) => {
      event.preventDefault();
      zone.classList.remove("dragover");
      const file = event.dataTransfer && event.dataTransfer.files[0];
      if (file) setFile(slot, file);
    });
  }

  function showError(message) {
    statusPanel.hidden = false;
    errorMsg.hidden = false;
    errorMsg.textContent = message;
  }

  function hideError() {
    errorMsg.hidden = true;
    errorMsg.textContent = "";
  }

  function resetResultUi() {
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }
    downloadBtn.hidden = true;
    downloadBtn.removeAttribute("href");
    progressWrap.hidden = false;
    progressBar.classList.remove("done");
    progressBar.style.width = "";
    progressBar.classList.add("indeterminate");
    hideError();
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function runMashup() {
    if (!songA || !songB) return;

    resetResultUi();
    statusPanel.hidden = false;
    mashupBtn.classList.add("busy");
    mashupBtn.disabled = true;
    mashupBtn.textContent = "Mixing…";

    let stageIndex = 0;
    progressLabel.textContent = STAGE_MESSAGES[0];
    const started = Date.now();
    elapsedEl.textContent = "0:00";

    const timers = [];
    timers.push(
      setInterval(() => {
        elapsedEl.textContent = formatElapsed(Date.now() - started);
      }, 250)
    );
    timers.push(
      setInterval(() => {
        stageIndex = Math.min(stageIndex + 1, STAGE_MESSAGES.length - 1);
        progressLabel.textContent = STAGE_MESSAGES[stageIndex];
      }, 18000)
    );

    const form = new FormData();
    form.append("song_a", songA, songA.name);
    form.append("song_b", songB, songB.name);

    try {
      const response = await fetch("/api/mashup", {
        method: "POST",
        body: form,
      });

      if (!response.ok) {
        let detail = `Request failed (${response.status})`;
        try {
          const payload = await response.json();
          if (payload && payload.detail) {
            detail =
              typeof payload.detail === "string"
                ? payload.detail
                : JSON.stringify(payload.detail);
          }
        } catch {
          /* ignore non-JSON error bodies */
        }
        throw new Error(detail);
      }

      const blob = await response.blob();
      objectUrl = URL.createObjectURL(blob);

      // Finish the bar before revealing download (and keep download hidden until then).
      timers.forEach(clearInterval);
      timers.length = 0;
      downloadBtn.hidden = true;
      progressWrap.hidden = false;
      progressBar.classList.remove("indeterminate");
      progressBar.classList.add("done");
      progressBar.style.width = "100%";
      progressLabel.textContent = "Ready";
      elapsedEl.textContent = formatElapsed(Date.now() - started);
      await sleep(450);

      progressWrap.hidden = true;
      downloadBtn.href = objectUrl;
      downloadBtn.hidden = false;
    } catch (err) {
      progressWrap.hidden = true;
      downloadBtn.hidden = true;
      const message =
        err instanceof Error ? err.message : "Something went wrong during mashup.";
      showError(message);
    } finally {
      timers.forEach(clearInterval);
      mashupBtn.classList.remove("busy");
      mashupBtn.textContent = "Mashup";
      updateButton();
    }
  }

  wireZone(zoneA, fileA, "a");
  wireZone(zoneB, fileB, "b");
  mashupBtn.addEventListener("click", runMashup);
})();
