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
  const resultRow = document.getElementById("result-row");
  const mashupAudio = document.getElementById("mashup-audio");
  const playerPlay = document.getElementById("player-play");
  const playerSeek = document.getElementById("player-seek");
  const playerTime = document.getElementById("player-time");
  const vocalPolicy = document.getElementById("vocal-policy");
  const creativeMode = document.getElementById("creative-mode");
  const structureMode = document.getElementById("structure-mode");
  const sectionEditor = document.getElementById("section-editor");
  const studioBoard = document.getElementById("studio-board");
  const studioPlayhead = document.getElementById("studio-playhead");
  const studioPlay = document.getElementById("studio-play");
  const studioDownload = document.getElementById("studio-download");
  const studioAudio = document.getElementById("studio-audio");
  const studioTime = document.getElementById("studio-time");
  const arrangementHint = document.getElementById("arrangement-hint");

  const STEMS = [
    { id: "vocals", label: "vocals" },
    { id: "bass", label: "bass" },
    { id: "drums", label: "percussion" },
    { id: "other", label: "other" },
  ];

  /** @type {File | null} */
  let songA = null;
  /** @type {File | null} */
  let songB = null;
  /** @type {string | null} */
  let objectUrl = null;
  /** @type {string | null} */
  let studioObjectUrl = null;
  /** @type {string | null} */
  let sessionId = null;
  /** @type {object | null} */
  let mashupMeta = null;
  /** @type {object | null} */
  let studioState = null;

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

  function formatTime(sec) {
    if (!Number.isFinite(sec) || sec < 0) return "0:00";
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
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
    errorMsg.hidden = false;
    errorMsg.textContent = message;
  }

  function hideError() {
    errorMsg.hidden = true;
    errorMsg.textContent = "";
  }

  function decodeMetadataHeader(header) {
    if (!header) return null;
    try {
      const json = atob(header);
      return JSON.parse(json);
    } catch {
      return null;
    }
  }

  function showMashupResult(url) {
    resultRow.hidden = false;
    downloadBtn.href = url;
    mashupAudio.src = url;
    mashupAudio.load();
    playerPlay.textContent = "Play";
    playerSeek.value = "0";
    playerSeek.max = "0";
    playerTime.textContent = "0:00 / 0:00";
  }

  function cellKey(song, stem) {
    return `${song}:${stem}`;
  }

  function sectionLabel(sections, index) {
    if (!sections || !sections.length) return `§${index}`;
    const sec = sections[index % sections.length];
    return sec.name || sec.label || `§${index}`;
  }

  function columnWidths() {
    if (!studioState || !studioState.columns) return [];
    return studioState.columns.map((col) => {
      const ms = Number(col.duration_ms) || 4000;
      return Math.max(96, Math.round(ms / 40));
    });
  }

  async function persistStudio() {
    if (!sessionId || !studioState) return;
    try {
      await fetch(`/api/mashup/sessions/${sessionId}/studio`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ studio: studioState }),
      });
    } catch {
      /* ignore transient save errors */
    }
  }

  function insertColumn(atIndex) {
    if (!studioState) return;
    const neighbor =
      studioState.columns[Math.max(0, atIndex - 1)] ||
      studioState.columns[0] ||
      null;
    const cells = {};
    for (const song of ["a", "b"]) {
      for (const stem of STEMS) {
        cells[cellKey(song, stem.id)] = {
          enabled: false,
          source_section_index: 0,
        };
      }
    }
    const col = {
      id: `col_${Date.now().toString(36)}`,
      label: `Section ${(studioState.columns || []).length + 1}`,
      duration_ms: neighbor ? neighbor.duration_ms : 8000,
      cells,
    };
    studioState.columns.splice(atIndex, 0, col);
    renderStudioGrid();
    persistStudio();
  }

  function openSectionPicker(song, stemId, colIndex, anchorEl) {
    const existing = document.querySelector(".studio-picker");
    if (existing) existing.remove();
    const sections =
      song === "a" ? studioState.sections_a || [] : studioState.sections_b || [];
    const picker = document.createElement("div");
    picker.className = "studio-picker";
    const off = document.createElement("button");
    off.type = "button";
    off.textContent = "Off";
    off.addEventListener("click", () => {
      const key = cellKey(song, stemId);
      studioState.columns[colIndex].cells[key] = {
        enabled: false,
        source_section_index: 0,
      };
      picker.remove();
      renderStudioGrid();
      persistStudio();
    });
    picker.appendChild(off);
    sections.forEach((sec, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = sec.name || sec.label || `Section ${i}`;
      btn.addEventListener("click", () => {
        const key = cellKey(song, stemId);
        studioState.columns[colIndex].cells[key] = {
          enabled: true,
          source_section_index: Number(sec.index != null ? sec.index : i),
        };
        picker.remove();
        renderStudioGrid();
        persistStudio();
      });
      picker.appendChild(btn);
    });
    document.body.appendChild(picker);
    const rect = anchorEl.getBoundingClientRect();
    picker.style.left = `${Math.min(rect.left, window.innerWidth - 200)}px`;
    picker.style.top = `${rect.bottom + 4}px`;
    const closer = (event) => {
      if (!picker.contains(event.target) && event.target !== anchorEl) {
        picker.remove();
        document.removeEventListener("mousedown", closer);
      }
    };
    setTimeout(() => document.addEventListener("mousedown", closer), 0);
  }

  function renderStudioGrid() {
    if (!studioBoard || !studioState) return;
    const widths = columnWidths();
    const totalWidth = widths.reduce((a, b) => a + b, 0) + 120;
    studioBoard.style.width = `${totalWidth}px`;
    studioBoard.innerHTML = "";
    if (studioPlayhead) {
      studioPlayhead.hidden = true;
      studioBoard.appendChild(studioPlayhead);
    }

    const header = document.createElement("div");
    header.className = "studio-row studio-header";
    const corner = document.createElement("div");
    corner.className = "studio-label";
    corner.textContent = "";
    header.appendChild(corner);
    studioState.columns.forEach((col, i) => {
      const cell = document.createElement("div");
      cell.className = "studio-col-head";
      cell.style.width = `${widths[i]}px`;
      cell.textContent = col.label || `Sec ${i + 1}`;
      const edge = document.createElement("button");
      edge.type = "button";
      edge.className = "studio-edge-add";
      edge.title = "Insert section";
      edge.textContent = "+";
      edge.addEventListener("click", (event) => {
        event.stopPropagation();
        insertColumn(i + 1);
      });
      cell.appendChild(edge);
      header.appendChild(cell);
    });
    studioBoard.appendChild(header);

    const leadingEdge = document.createElement("button");
    leadingEdge.type = "button";
    leadingEdge.className = "studio-edge-add leading";
    leadingEdge.textContent = "+";
    leadingEdge.title = "Insert section at start";
    leadingEdge.addEventListener("click", () => insertColumn(0));
    corner.appendChild(leadingEdge);

    for (const song of [
      { id: "a", title: studioState.title_a || "Song A" },
      { id: "b", title: studioState.title_b || "Song B" },
    ]) {
      const group = document.createElement("div");
      group.className = "studio-group";
      const groupLabel = document.createElement("div");
      groupLabel.className = "studio-group-label";
      groupLabel.textContent = song.title;
      group.appendChild(groupLabel);

      for (const stem of STEMS) {
        const row = document.createElement("div");
        row.className = "studio-row";
        const label = document.createElement("div");
        label.className = "studio-label";
        label.textContent = stem.label;
        row.appendChild(label);

        studioState.columns.forEach((col, colIndex) => {
          const key = cellKey(song.id, stem.id);
          const data = (col.cells && col.cells[key]) || {
            enabled: false,
            source_section_index: 0,
          };
          const cell = document.createElement("button");
          cell.type = "button";
          cell.className = `studio-cell${data.enabled ? " on" : " off"}`;
          cell.style.width = `${widths[colIndex]}px`;
          const sections =
            song.id === "a"
              ? studioState.sections_a || []
              : studioState.sections_b || [];
          cell.textContent = data.enabled
            ? sectionLabel(sections, data.source_section_index)
            : "—";
          cell.addEventListener("click", () =>
            openSectionPicker(song.id, stem.id, colIndex, cell)
          );
          row.appendChild(cell);
        });
        group.appendChild(row);
      }
      studioBoard.appendChild(group);
    }
    sectionEditor.hidden = false;
  }

  async function loadStudioState() {
    if (!sessionId) return;
    const response = await fetch(`/api/mashup/sessions/${sessionId}/studio`);
    if (!response.ok) throw new Error("Failed to load section editor");
    studioState = await response.json();
    renderStudioGrid();
  }

  function updateArrangementHint(meta) {
    if (!arrangementHint) return;
    const parts = [];
    if (meta.structure_mode) parts.push(`structure ${meta.structure_mode}`);
    if (meta.structure_source) parts.push(`source ${meta.structure_source}`);
    if (meta.arranging_reasoning) parts.push(meta.arranging_reasoning);
    if (Array.isArray(meta.phrases)) parts.push(`${meta.phrases.length} sections`);
    if (parts.length) {
      arrangementHint.hidden = false;
      arrangementHint.textContent = parts.join(" · ");
    } else {
      arrangementHint.hidden = true;
    }
  }

  async function hydrateSession(id) {
    sessionId = id;
    statusPanel.hidden = false;
    progressWrap.hidden = true;
    hideError();
    const metaResp = await fetch(`/api/mashup/sessions/${id}`);
    if (!metaResp.ok) throw new Error("Session not found");
    mashupMeta = await metaResp.json();
    mashupMeta.session_id = id;
    updateArrangementHint(mashupMeta);
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = `/api/mashup/sessions/${id}/mashup`;
    showMashupResult(objectUrl);
    await loadStudioState();
  }

  async function playStudioEdit() {
    if (!sessionId || !studioPlay) return;
    studioPlay.disabled = true;
    studioPlay.textContent = "Rendering…";
    try {
      await persistStudio();
      const response = await fetch(
        `/api/mashup/sessions/${sessionId}/studio/preview`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        }
      );
      if (!response.ok) {
        let detail = `Preview failed (${response.status})`;
        try {
          const payload = await response.json();
          if (payload && payload.detail) detail = String(payload.detail);
        } catch {
          /* ignore */
        }
        throw new Error(detail);
      }
      const blob = await response.blob();
      if (studioObjectUrl) URL.revokeObjectURL(studioObjectUrl);
      studioObjectUrl = URL.createObjectURL(blob);
      studioAudio.src = studioObjectUrl;
      studioAudio.load();
      await studioAudio.play();
      studioPlay.textContent = "Pause edit";
      if (studioDownload) {
        studioDownload.hidden = false;
        studioDownload.href = studioObjectUrl;
      }
      if (studioPlayhead) studioPlayhead.hidden = false;
    } catch (err) {
      showError(err instanceof Error ? err.message : "Studio preview failed.");
      studioPlay.textContent = "Play edit";
    } finally {
      studioPlay.disabled = false;
    }
  }

  function updateStudioPlayhead() {
    if (!studioPlayhead || !studioState || !studioAudio.duration) return;
    const widths = columnWidths();
    const totalMs = studioState.columns.reduce(
      (sum, col) => sum + (Number(col.duration_ms) || 0),
      0
    );
    const t = studioAudio.currentTime;
    const ratio = totalMs > 0 ? (t * 1000) / totalMs : t / studioAudio.duration;
    const x = 120 + ratio * widths.reduce((a, b) => a + b, 0);
    studioPlayhead.style.transform = `translateX(${x}px)`;
    studioPlayhead.hidden = false;
    if (studioTime) {
      studioTime.textContent = `${formatTime(t)} / ${formatTime(
        studioAudio.duration || 0
      )}`;
    }
  }

  async function runMashup() {
    if (!songA || !songB) return;
    mashupBtn.classList.add("busy");
    mashupBtn.disabled = true;
    statusPanel.hidden = false;
    progressWrap.hidden = false;
    resultRow.hidden = true;
    sectionEditor.hidden = true;
    hideError();

    const started = Date.now();
    let stage = 0;
    progressLabel.textContent = STAGE_MESSAGES[0];
    progressBar.style.width = "8%";
    const ticker = setInterval(() => {
      elapsedEl.textContent = formatElapsed(Date.now() - started);
      stage = Math.min(stage + 1, STAGE_MESSAGES.length - 1);
      progressLabel.textContent = STAGE_MESSAGES[stage];
      const pct = Math.min(92, 8 + stage * 16);
      progressBar.style.width = `${pct}%`;
    }, 4000);

    const form = new FormData();
    form.append("song_a", songA);
    form.append("song_b", songB);
    form.append("vocal_policy", vocalPolicy.value);
    form.append("creative_mode", creativeMode.value);
    form.append(
      "structure_mode",
      structureMode ? structureMode.value : "allin1"
    );

    try {
      const response = await fetch("/api/mashup", { method: "POST", body: form });
      if (!response.ok) {
        let detail = `Mashup failed (${response.status})`;
        try {
          const payload = await response.json();
          if (payload && payload.detail) detail = String(payload.detail);
        } catch {
          /* ignore */
        }
        throw new Error(detail);
      }
      sessionId = response.headers.get("X-Mashup-Session-Id");
      mashupMeta = decodeMetadataHeader(
        response.headers.get("X-Mashup-Metadata")
      );
      if (mashupMeta && sessionId) mashupMeta.session_id = sessionId;
      const blob = await response.blob();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = URL.createObjectURL(blob);
      progressBar.style.width = "100%";
      progressLabel.textContent = "Done";
      progressWrap.hidden = true;
      showMashupResult(objectUrl);
      if (mashupMeta) updateArrangementHint(mashupMeta);
      if (sessionId) {
        const url = new URL(window.location.href);
        url.searchParams.set("session", sessionId);
        window.history.replaceState({}, "", url);
        await loadStudioState();
      }
    } catch (err) {
      progressWrap.hidden = true;
      showError(err instanceof Error ? err.message : "Mashup failed.");
    } finally {
      clearInterval(ticker);
      elapsedEl.textContent = formatElapsed(Date.now() - started);
      mashupBtn.classList.remove("busy");
      updateButton();
    }
  }

  wireZone(zoneA, fileA, "a");
  wireZone(zoneB, fileB, "b");
  mashupBtn.addEventListener("click", runMashup);

  playerPlay.addEventListener("click", async () => {
    if (mashupAudio.paused) {
      await mashupAudio.play();
      playerPlay.textContent = "Pause";
    } else {
      mashupAudio.pause();
      playerPlay.textContent = "Play";
    }
  });
  mashupAudio.addEventListener("loadedmetadata", () => {
    playerSeek.max = String(mashupAudio.duration || 0);
    playerTime.textContent = `0:00 / ${formatTime(mashupAudio.duration || 0)}`;
  });
  mashupAudio.addEventListener("timeupdate", () => {
    playerSeek.value = String(mashupAudio.currentTime);
    playerTime.textContent = `${formatTime(mashupAudio.currentTime)} / ${formatTime(
      mashupAudio.duration || 0
    )}`;
  });
  mashupAudio.addEventListener("ended", () => {
    playerPlay.textContent = "Play";
  });
  playerSeek.addEventListener("input", () => {
    mashupAudio.currentTime = Number(playerSeek.value);
  });

  if (studioPlay) {
    studioPlay.addEventListener("click", async () => {
      if (!studioAudio.paused && studioAudio.src) {
        studioAudio.pause();
        studioPlay.textContent = "Play edit";
        return;
      }
      await playStudioEdit();
    });
  }
  studioAudio.addEventListener("timeupdate", updateStudioPlayhead);
  studioAudio.addEventListener("ended", () => {
    if (studioPlay) studioPlay.textContent = "Play edit";
  });
  if (studioDownload) {
    studioDownload.addEventListener("click", async (event) => {
      if (!sessionId) return;
      if (studioDownload.dataset.ready === "1") {
        studioDownload.dataset.ready = "";
        return;
      }
      event.preventDefault();
      try {
        await persistStudio();
        const response = await fetch(
          `/api/mashup/sessions/${sessionId}/studio/render`,
          { method: "POST" }
        );
        if (!response.ok) throw new Error("Render failed");
        const blob = await response.blob();
        if (studioObjectUrl) URL.revokeObjectURL(studioObjectUrl);
        studioObjectUrl = URL.createObjectURL(blob);
        studioDownload.href = studioObjectUrl;
        studioDownload.dataset.ready = "1";
        studioDownload.click();
      } catch (err) {
        showError(err instanceof Error ? err.message : "Download edit failed.");
      }
    });
  }

  const params = new URLSearchParams(window.location.search);
  const hydrateId = params.get("session");
  if (hydrateId) {
    hydrateSession(hydrateId).catch((err) => {
      showError(err instanceof Error ? err.message : "Failed to restore session");
      statusPanel.hidden = false;
    });
  }
})();
