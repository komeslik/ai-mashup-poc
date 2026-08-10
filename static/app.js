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
  const vocalPolicy = document.getElementById("vocal-policy");
  const creativeMode = document.getElementById("creative-mode");
  const sectionEditor = document.getElementById("section-editor");
  const phraseList = document.getElementById("phrase-list");
  const rebuildBtn = document.getElementById("rebuild-btn");
  const libraryList = document.getElementById("library-list");
  const libraryRefresh = document.getElementById("library-refresh");
  const librarySearch = document.getElementById("library-search");

  /** @type {File | null} */
  let songA = null;
  /** @type {File | null} */
  let songB = null;
  /** @type {string | null} */
  let objectUrl = null;
  /** @type {string | null} */
  let sessionId = null;
  /** @type {object | null} */
  let mashupMeta = null;

  const STAGE_MESSAGES = [
    "Uploading tracks…",
    "Separating stems with local Demucs…",
    "Detecting sections + DJ arrangement…",
    "Building contiguous section mashup…",
    "Mixing + mastering…",
    "Almost done…",
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
    if (librarySearch) librarySearch.disabled = !songA;
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
    if (sectionEditor) sectionEditor.hidden = true;
    if (rebuildBtn) rebuildBtn.hidden = true;
    if (phraseList) phraseList.innerHTML = "";
    sessionId = null;
    mashupMeta = null;
    progressWrap.hidden = false;
    progressBar.classList.remove("done");
    progressBar.style.width = "";
    progressBar.classList.add("indeterminate");
    hideError();
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function decodeMetadataHeader(value) {
    if (!value) return null;
    try {
      const json = atob(value);
      return JSON.parse(json);
    } catch {
      return null;
    }
  }

  function renderSectionEditor(meta) {
    if (!sectionEditor || !phraseList || !meta || !Array.isArray(meta.phrases)) {
      return;
    }
    const directorHint = document.getElementById("director-mode-hint");
    if (directorHint) {
      const strict = meta.director_strict !== false;
      directorHint.hidden = false;
      directorHint.textContent = strict
        ? "AI Director · strict (one vocal at a time)"
        : "AI Director · muted harmony allowed (overlap hard-muted)";
    }
    const arrangementHint = document.getElementById("arrangement-hint");
    if (arrangementHint) {
      if (meta.arranging_reasoning) {
        arrangementHint.hidden = false;
        arrangementHint.textContent = meta.arranging_reasoning;
      } else {
        arrangementHint.hidden = true;
        arrangementHint.textContent = "";
      }
    }
    const stemHint = document.getElementById("stem-actions-hint");
    if (stemHint) {
      const actions = Array.isArray(meta.stem_actions) ? meta.stem_actions : [];
      if (actions.length) {
        stemHint.hidden = false;
        stemHint.textContent = actions
          .map((a) => {
            const bars = `${a.bar_start}–${a.bar_end}`;
            const vox = a.vocal_source === "none" ? "instr" : a.vocal_source;
            return `bars ${bars}: ${vox}`;
          })
          .join(" · ");
      } else {
        stemHint.hidden = true;
        stemHint.textContent = "";
      }
    }
    phraseList.innerHTML = "";
    meta.phrases.forEach((phrase) => {
      const li = document.createElement("li");
      li.className = "phrase-item";
      const checked = phrase.enabled !== false ? "checked" : "";
      const title = phrase.section_name || `S${phrase.index}`;
      const role = phrase.label ? ` · ${phrase.label}` : "";
      li.innerHTML = `
        <label>
          <input type="checkbox" data-phrase-index="${phrase.index}" ${checked} />
          <span class="phrase-main">${title}${role} · ${phrase.lead}</span>
          <span class="phrase-meta">
            score ${Number(phrase.mashability_score || 0).toFixed(2)}
            · rhythm ${Number(phrase.rhythmic_score || 0).toFixed(2)}
            · ${phrase.n_steps >= 0 ? "+" : ""}${phrase.n_steps}st
            ${phrase.harmony ? "· muted harmony" : ""}
            ${phrase.vad_muted_frames ? `· VAD mute ${phrase.vad_muted_frames}` : ""}
            ${phrase.instr_ducked_frames ? `· instr duck ${phrase.instr_ducked_frames}` : ""}
          </span>
        </label>
      `;
      phraseList.appendChild(li);
    });
    sectionEditor.hidden = false;
    if (rebuildBtn) rebuildBtn.hidden = false;
  }

  async function rebuildFromSelection() {
    if (!sessionId || !rebuildBtn) return;
    const boxes = phraseList.querySelectorAll('input[type="checkbox"]');
    const enabled = [];
    boxes.forEach((box) => {
      if (box.checked) enabled.push(Number(box.dataset.phraseIndex));
    });
    if (!enabled.length) {
      showError("Enable at least one phrase to rebuild.");
      return;
    }

    rebuildBtn.disabled = true;
    rebuildBtn.textContent = "Rebuilding…";
    hideError();
    try {
      const response = await fetch(`/api/mashup/sessions/${sessionId}/reassemble`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled_indices: enabled }),
      });
      if (!response.ok) {
        let detail = `Rebuild failed (${response.status})`;
        try {
          const payload = await response.json();
          if (payload && payload.detail) detail = String(payload.detail);
        } catch {
          /* ignore */
        }
        throw new Error(detail);
      }
      const blob = await response.blob();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = URL.createObjectURL(blob);
      downloadBtn.href = objectUrl;
      downloadBtn.hidden = false;
      const sessionResp = await fetch(`/api/mashup/sessions/${sessionId}`);
      if (sessionResp.ok) {
        mashupMeta = await sessionResp.json();
        mashupMeta.session_id = sessionId;
        renderSectionEditor(mashupMeta);
      }
    } catch (err) {
      showError(err instanceof Error ? err.message : "Rebuild failed.");
    } finally {
      rebuildBtn.disabled = false;
      rebuildBtn.textContent = "Rebuild from selection";
    }
  }

  async function refreshLibrary() {
    if (!libraryList) return;
    libraryList.innerHTML = "<li class='library-item'>Loading…</li>";
    try {
      const response = await fetch("/api/library");
      const payload = await response.json();
      const tracks = payload.tracks || [];
      if (!tracks.length) {
        libraryList.innerHTML =
          "<li class='library-item'>No tracks yet — add files under library/</li>";
        return;
      }
      libraryList.innerHTML = "";
      tracks.forEach((track) => {
        const li = document.createElement("li");
        li.className = "library-item";
        li.textContent = `${track.name}${track.bpm ? ` · ${track.bpm.toFixed(1)} BPM` : ""}`;
        libraryList.appendChild(li);
      });
    } catch (err) {
      libraryList.innerHTML = `<li class='library-item'>Failed to load library</li>`;
    }
  }

  async function searchLibrary() {
    if (!songA || !libraryList) return;
    libraryList.innerHTML = "<li class='library-item'>Ranking…</li>";
    const form = new FormData();
    form.append("query", songA, songA.name);
    form.append("top_k", "5");
    try {
      const response = await fetch("/api/library/search", { method: "POST", body: form });
      const payload = await response.json();
      const results = payload.results || [];
      if (!results.length) {
        libraryList.innerHTML =
          "<li class='library-item'>No ranked matches (add more library tracks)</li>";
        return;
      }
      libraryList.innerHTML = "";
      results.forEach((row) => {
        const li = document.createElement("li");
        li.className = "library-item";
        li.textContent = `${row.name} · score ${Number(row.score).toFixed(3)} · ${row.n_steps >= 0 ? "+" : ""}${row.n_steps}st`;
        libraryList.appendChild(li);
      });
    } catch {
      libraryList.innerHTML = "<li class='library-item'>Library search failed</li>";
    }
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
    form.append("vocal_policy", vocalPolicy ? vocalPolicy.value : "auto");
    form.append("creative_mode", creativeMode ? creativeMode.value : "forced_match");

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

      sessionId = response.headers.get("X-Mashup-Session-Id");
      mashupMeta = decodeMetadataHeader(response.headers.get("X-Mashup-Metadata"));
      if (mashupMeta && sessionId) mashupMeta.session_id = sessionId;

      const blob = await response.blob();
      objectUrl = URL.createObjectURL(blob);

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
      if (mashupMeta) renderSectionEditor(mashupMeta);
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
  if (rebuildBtn) rebuildBtn.addEventListener("click", rebuildFromSelection);
  if (libraryRefresh) libraryRefresh.addEventListener("click", refreshLibrary);
  if (librarySearch) librarySearch.addEventListener("click", searchLibrary);
  refreshLibrary();
})();
