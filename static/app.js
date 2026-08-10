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
  const studioSkip = document.getElementById("studio-skip");
  const studioDownload = document.getElementById("studio-download");
  const studioAudio = document.getElementById("studio-audio");
  const sectionPreviewAudio = document.getElementById("section-preview-audio");
  const studioTime = document.getElementById("studio-time");
  const arrangementHint = document.getElementById("arrangement-hint");
  const commitSectionsBtn = document.getElementById("commit-sections");
  const studioWaveforms = document.getElementById("studio-waveforms");
  const LABEL_W = 120;
  const PX_PER_SEC = 40;
  const MIN_SECTION_SEC = 0.25;

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
  let sectionPreviewUrl = null;
  /** @type {{ song: string, index: number, start: number, end: number } | null} */
  let activeSectionPreview = null;
  /** @type {string | null} */
  let sessionId = null;
  /** @type {object | null} */
  let mashupMeta = null;
  /** @type {object | null} */
  let studioState = null;
  let studioDirty = true;
  let studioAudioReady = false;
  let playheadTimeSec = 0;
  let studioBusy = false;
  let studioRerenderToken = 0;

  /** @type {{ a: object[] | null, b: object[] | null }} */
  let pendingSections = { a: null, b: null };
  /** @type {{ a: Float32Array | null, b: Float32Array | null, aDur: number, bDur: number }} */
  let wavePeaks = { a: null, b: null, aDur: 0, bDur: 0 };

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

  function cloneSections(sections) {
    return (sections || []).map((s) => ({ ...s }));
  }

  function sectionsForSong(song) {
    if (song === "a") {
      return pendingSections.a || studioState?.sections_a || [];
    }
    return pendingSections.b || studioState?.sections_b || [];
  }

  function sectionDisplayLabel(sec, index) {
    if (!sec) return `§${index}`;
    return sec.display_label || sec.name || sec.label || `§${index}`;
  }

  function sectionLabel(sections, index) {
    if (!sections || !sections.length) return `§${index}`;
    const sec = sections[index % sections.length];
    return sectionDisplayLabel(sec, index);
  }

  function sectionColor(sections, index) {
    if (!sections || !sections.length) return "#6ee7b7";
    const sec = sections[index % sections.length];
    return sec.color || "#6ee7b7";
  }

  function sectionDurationMs(sections, index) {
    if (!sections || !sections.length) return 4000;
    const sec = sections[index % sections.length];
    const start = Number(sec.start_sec) || 0;
    const end = Number(sec.end_sec) || start + 4;
    return Math.max(250, Math.round((end - start) * 1000));
  }

  function columnEffectiveDurationMs(col) {
    if (!studioState || !col) return Number(col?.duration_ms) || 4000;
    const sectionsA = sectionsForSong("a");
    const sectionsB = sectionsForSong("b");
    const lengths = [];
    const cells = col.cells || {};
    for (const song of ["a", "b"]) {
      const sections = song === "a" ? sectionsA : sectionsB;
      for (const stem of STEMS) {
        const data = cells[cellKey(song, stem.id)];
        if (!data || !data.enabled) continue;
        lengths.push(
          sectionDurationMs(sections, Number(data.source_section_index) || 0)
        );
      }
    }
    const stored = Number(col.duration_ms) || 0;
    if (lengths.length) {
      const natural = Math.max(...lengths);
      return stored > 0 ? Math.max(stored, natural) : natural;
    }
    return stored || 4000;
  }

  function totalTimelineMs() {
    if (!studioState?.columns) return 0;
    return studioState.columns.reduce(
      (sum, col) => sum + columnEffectiveDurationMs(col),
      0
    );
  }

  function columnWidths() {
    if (!studioState || !studioState.columns) return [];
    return studioState.columns.map((col) => {
      const ms = columnEffectiveDurationMs(col);
      return Math.max(96, Math.round(ms / 40));
    });
  }

  function markStudioDirty() {
    studioDirty = true;
    studioAudioReady = false;
  }

  function setRenderingUi(on) {
    if (!studioPlay) return;
    studioPlay.classList.toggle("rendering", on);
    const playIcon = studioPlay.querySelector(".icon-play");
    const pauseIcon = studioPlay.querySelector(".icon-pause");
    const spinner = studioPlay.querySelector(".icon-spinner");
    if (spinner) spinner.hidden = !on;
    if (on) {
      if (playIcon) playIcon.hidden = true;
      if (pauseIcon) pauseIcon.hidden = true;
      studioPlay.title = "Rendering…";
      studioPlay.setAttribute("aria-label", "Rendering");
      studioPlay.disabled = true;
    } else {
      studioPlay.disabled = false;
      setPlayIcons(!studioAudio.paused && !!studioAudio.src);
    }
  }

  function setPlayIcons(playing) {
    if (!studioPlay) return;
    if (studioPlay.classList.contains("rendering")) return;
    const playIcon = studioPlay.querySelector(".icon-play");
    const pauseIcon = studioPlay.querySelector(".icon-pause");
    const spinner = studioPlay.querySelector(".icon-spinner");
    if (spinner) spinner.hidden = true;
    if (playIcon) playIcon.hidden = playing;
    if (pauseIcon) pauseIcon.hidden = !playing;
    studioPlay.title = playing ? "Pause" : "Play";
    studioPlay.setAttribute("aria-label", playing ? "Pause" : "Play");
  }

  /**
   * After a grid cell/column edit: freeze playhead, pause, show spinner,
   * re-render preview, then resume from the frozen ticker if it was playing.
   */
  async function afterStudioGridEdit() {
    if (!sessionId || !studioState) return;
    if (studioAudio.src && Number.isFinite(studioAudio.currentTime)) {
      playheadTimeSec = studioAudio.currentTime;
    }
    const resumeAfter = Boolean(studioAudio.src && !studioAudio.paused);
    studioAudio.pause();
    setPlayIcons(false);
    setPlayheadVisual(playheadTimeSec);

    markStudioDirty();
    const token = ++studioRerenderToken;
    setRenderingUi(true);
    studioBusy = true;
    try {
      await persistStudio();
      await ensureStudioPreviewLoaded();
      if (token !== studioRerenderToken) return;
      const max = studioAudio.duration || 0;
      const seekTo = Math.max(0, Math.min(playheadTimeSec, max || playheadTimeSec));
      studioAudio.currentTime = seekTo;
      playheadTimeSec = seekTo;
      setPlayheadVisual(playheadTimeSec);
      if (resumeAfter) {
        await studioAudio.play();
        setPlayIcons(true);
      }
    } catch (err) {
      if (token === studioRerenderToken) {
        showError(err instanceof Error ? err.message : "Studio preview failed.");
        setPlayIcons(false);
      }
    } finally {
      if (token === studioRerenderToken) {
        setRenderingUi(false);
        studioBusy = false;
      }
    }
  }

  function setPlayheadVisual(timeSec) {
    if (!studioPlayhead || !studioState) return;
    const widths = columnWidths();
    const totalMs = totalTimelineMs();
    const totalW = widths.reduce((a, b) => a + b, 0);
    const ratio = totalMs > 0 ? (timeSec * 1000) / totalMs : 0;
    const x = LABEL_W + Math.max(0, Math.min(1, ratio)) * totalW;
    studioPlayhead.style.transform = `translateX(${x}px)`;
    studioPlayhead.hidden = false;
    const dur =
      studioAudioReady && studioAudio.duration
        ? studioAudio.duration
        : totalMs / 1000;
    if (studioTime) {
      studioTime.textContent = `${formatTime(timeSec)} / ${formatTime(dur || 0)}`;
    }
  }

  function seekPlayheadToTime(timeSec, { resumeIfPlaying = true } = {}) {
    const totalMs = totalTimelineMs();
    const maxSec =
      studioAudioReady && studioAudio.duration
        ? studioAudio.duration
        : totalMs / 1000;
    const t = Math.max(0, Math.min(maxSec || 0, timeSec));
    playheadTimeSec = t;
    setPlayheadVisual(t);
    if (studioAudioReady && studioAudio.src) {
      const wasPlaying = !studioAudio.paused;
      studioAudio.currentTime = t;
      if (resumeIfPlaying && wasPlaying) {
        studioAudio.play().catch(() => {});
      }
    }
  }

  function timeFromClientX(clientX) {
    if (!studioBoard) return 0;
    const rect = studioBoard.getBoundingClientRect();
    const x = clientX - rect.left - LABEL_W;
    const widths = columnWidths();
    const totalW = widths.reduce((a, b) => a + b, 0);
    if (totalW <= 0) return 0;
    const clamped = Math.max(0, Math.min(totalW, x));
    const totalMs = totalTimelineMs();
    return (clamped / totalW) * (totalMs / 1000);
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
    afterStudioGridEdit();
  }

  function removeColumn(index) {
    if (!studioState || studioState.columns.length <= 1) return;
    studioState.columns.splice(index, 1);
    renderStudioGrid();
    afterStudioGridEdit();
  }

  function openSectionPicker(song, stemId, colIndex, anchorEl) {
    const existing = document.querySelector(".studio-picker");
    if (existing) existing.remove();
    const sections = sectionsForSong(song);
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
      afterStudioGridEdit();
    });
    picker.appendChild(off);
    sections.forEach((sec, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.background = sec.color || "#6ee7b7";
      btn.appendChild(swatch);
      btn.appendChild(
        document.createTextNode(sectionDisplayLabel(sec, i))
      );
      btn.addEventListener("click", () => {
        const key = cellKey(song, stemId);
        studioState.columns[colIndex].cells[key] = {
          enabled: true,
          source_section_index: Number(sec.index != null ? sec.index : i),
        };
        picker.remove();
        renderStudioGrid();
        afterStudioGridEdit();
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
    const totalWidth = widths.reduce((a, b) => a + b, 0) + LABEL_W;
    studioBoard.style.width = `${totalWidth}px`;
    studioBoard.innerHTML = "";
    if (studioPlayhead) {
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
      const title = document.createElement("span");
      title.textContent = col.label || `Sec ${i + 1}`;
      cell.appendChild(title);

      const controls = document.createElement("div");
      controls.className = "studio-edge-controls";
      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "studio-edge-remove";
      removeBtn.title = "Remove section";
      removeBtn.textContent = "−";
      removeBtn.disabled = studioState.columns.length <= 1;
      removeBtn.addEventListener("click", (event) => {
        event.stopPropagation();
        removeColumn(i);
      });
      const addBtn = document.createElement("button");
      addBtn.type = "button";
      addBtn.className = "studio-edge-add";
      addBtn.title = "Insert section";
      addBtn.textContent = "+";
      addBtn.addEventListener("click", (event) => {
        event.stopPropagation();
        insertColumn(i + 1);
      });
      controls.appendChild(removeBtn);
      controls.appendChild(addBtn);
      cell.appendChild(controls);
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
      groupLabel.style.width = `${totalWidth}px`;
      groupLabel.textContent = song.title;
      groupLabel.addEventListener("click", (event) => {
        const t = timeFromClientX(event.clientX);
        seekPlayheadToTime(t, { resumeIfPlaying: true });
      });
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
          const sections = sectionsForSong(song.id);
          if (data.enabled) {
            cell.textContent = sectionLabel(sections, data.source_section_index);
            cell.style.boxShadow = `inset 3px 0 0 ${sectionColor(
              sections,
              data.source_section_index
            )}`;
          } else {
            cell.textContent = "—";
            cell.style.boxShadow = "";
          }
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
    setPlayheadVisual(playheadTimeSec);
  }

  function updateCommitButton() {
    if (!commitSectionsBtn) return;
    const dirty =
      pendingSections.a !== null || pendingSections.b !== null;
    commitSectionsBtn.hidden = !dirty;
  }

  function songDuration(song) {
    if (song === "a") return wavePeaks.aDur || 0;
    return wavePeaks.bDur || 0;
  }

  function drawWaveform(song) {
    const canvas = document.getElementById(`wave-canvas-${song}`);
    const track = document.getElementById(`wave-track-${song}`);
    const peaks = wavePeaks[song];
    const dur = songDuration(song);
    if (!canvas || !track || !peaks || !dur) return;
    const width = Math.max(400, Math.round(dur * PX_PER_SEC));
    track.style.width = `${width}px`;
    canvas.width = width;
    canvas.height = 64;
    canvas.style.width = `${width}px`;
    canvas.style.height = "64px";
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, 64);
    ctx.fillStyle = "rgba(214, 245, 120, 0.08)";
    ctx.fillRect(0, 0, width, 64);
    ctx.fillStyle = "rgba(214, 245, 120, 0.55)";
    const mid = 32;
    for (let x = 0; x < width; x += 1) {
      const idx = Math.floor((x / width) * peaks.length);
      const amp = peaks[idx] || 0;
      const h = Math.max(1, amp * 28);
      ctx.fillRect(x, mid - h, 1, h * 2);
    }
  }

  function renderWaveRegions(song) {
    const host = document.getElementById(`wave-regions-${song}`);
    const track = document.getElementById(`wave-track-${song}`);
    if (!host || !track) return;
    const savedProgress =
      activeSectionPreview && activeSectionPreview.song === song
        ? sectionPreviewAudio.currentTime || 0
        : 0;
    host.innerHTML = "";
    const sections = sectionsForSong(song);
    const dur =
      songDuration(song) ||
      Math.max(...sections.map((s) => Number(s.end_sec) || 0), 1);
    const width = Math.max(400, Math.round(dur * PX_PER_SEC));
    track.style.width = `${width}px`;

    sections.forEach((sec, i) => {
      const start = Number(sec.start_sec) || 0;
      const end = Number(sec.end_sec) || start + 1;
      const left = (start / dur) * width;
      const w = Math.max(4, ((end - start) / dur) * width);
      const region = document.createElement("div");
      region.className = "wave-region";
      region.dataset.song = song;
      region.dataset.index = String(i);
      region.style.left = `${left}px`;
      region.style.width = `${w}px`;
      region.style.setProperty("--sec-color", sec.color || "#6ee7b7");

      const wrap = document.createElement("div");
      wrap.className = "wave-play-wrap";
      const nameEl = document.createElement("span");
      nameEl.className = "wave-sec-name";
      nameEl.textContent = sectionDisplayLabel(sec, i);
      const playBtn = document.createElement("button");
      playBtn.type = "button";
      playBtn.className = "wave-play";
      const isActive =
        activeSectionPreview &&
        activeSectionPreview.song === song &&
        activeSectionPreview.index === i;
      const isPlaying = isActive && !sectionPreviewAudio.paused;
      playBtn.textContent = isPlaying ? "❚❚" : "▶";
      playBtn.title = isPlaying ? "Pause section" : "Play section";
      playBtn.setAttribute("aria-label", playBtn.title);
      playBtn.addEventListener("click", (event) => {
        event.stopPropagation();
        toggleSectionPreview(song, i);
      });
      wrap.appendChild(nameEl);
      wrap.appendChild(playBtn);
      region.appendChild(wrap);

      const localHead = document.createElement("div");
      localHead.className = "wave-section-playhead";
      localHead.hidden = !(isActive && (isPlaying || savedProgress > 0.01));
      if (isActive) {
        const secDur = Math.max(0.05, end - start);
        const ratio = Math.max(0, Math.min(1, savedProgress / secDur));
        localHead.style.left = `${ratio * 100}%`;
      }
      region.appendChild(localHead);
      host.appendChild(region);
    });

    // Shared boundaries between adjacent sections.
    for (let i = 1; i < sections.length; i += 1) {
      const boundary = Number(sections[i].start_sec) || 0;
      host.appendChild(
        makeWaveHandle(song, (boundary / dur) * width, () =>
          beginBoundaryDrag(song, i - 1, i)
        )
      );
    }

    // Start of first section + end of last section.
    if (sections.length) {
      const firstStart = Number(sections[0].start_sec) || 0;
      const lastEnd =
        Number(sections[sections.length - 1].end_sec) || dur;
      host.appendChild(
        makeWaveHandle(song, (firstStart / dur) * width, () =>
          beginEdgeDrag(song, "start")
        )
      );
      host.appendChild(
        makeWaveHandle(song, (lastEnd / dur) * width, () =>
          beginEdgeDrag(song, "end")
        )
      );
    }
  }

  function makeWaveHandle(song, leftPx, onDown) {
    const handle = document.createElement("div");
    handle.className = "wave-handle";
    handle.style.left = `${leftPx}px`;
    handle.title = "Drag to resize sections";
    handle.setAttribute("role", "slider");
    handle.setAttribute("aria-label", "Section boundary");
    handle.addEventListener("mousedown", (event) => {
      event.preventDefault();
      event.stopPropagation();
      onDown();
    });
    void song;
    return handle;
  }

  function wireWaveformScrollSync() {
    const a = document.getElementById("wave-scroll-a");
    const b = document.getElementById("wave-scroll-b");
    if (!a || !b || a.dataset.syncBound === "1") return;
    a.dataset.syncBound = "1";
    b.dataset.syncBound = "1";
    let locking = false;
    const sync = (from, to) => {
      from.addEventListener("scroll", () => {
        if (locking) return;
        locking = true;
        to.scrollLeft = from.scrollLeft;
        locking = false;
      });
    };
    sync(a, b);
    sync(b, a);
  }

  function ensurePending(song) {
    if (song === "a") {
      if (!pendingSections.a) {
        pendingSections.a = cloneSections(studioState.sections_a);
      }
      return pendingSections.a;
    }
    if (!pendingSections.b) {
      pendingSections.b = cloneSections(studioState.sections_b);
    }
    return pendingSections.b;
  }

  function beginBoundaryDrag(song, leftIdx, rightIdx) {
    const sections = ensurePending(song);
    const dur = songDuration(song) || 1;
    const width = Math.max(400, Math.round(dur * PX_PER_SEC));
    const left = sections[leftIdx];
    const right = sections[rightIdx];
    const minT = (Number(left.start_sec) || 0) + MIN_SECTION_SEC;
    const maxT = (Number(right.end_sec) || dur) - MIN_SECTION_SEC;
    const onMove = (event) => {
      const track = document.getElementById(`wave-track-${song}`);
      if (!track) return;
      const rect = track.getBoundingClientRect();
      const x = event.clientX - rect.left;
      let t = (x / width) * dur;
      t = Math.max(minT, Math.min(maxT, t));
      left.end_sec = t;
      right.start_sec = t;
      renderWaveRegions(song);
      updateCommitButton();
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }

  function beginEdgeDrag(song, which) {
    const sections = ensurePending(song);
    if (!sections.length) return;
    const dur = songDuration(song) || 1;
    const width = Math.max(400, Math.round(dur * PX_PER_SEC));
    const onMove = (event) => {
      const track = document.getElementById(`wave-track-${song}`);
      if (!track) return;
      const rect = track.getBoundingClientRect();
      const x = event.clientX - rect.left;
      let t = (x / width) * dur;
      t = Math.max(0, Math.min(dur, t));
      if (which === "start") {
        const first = sections[0];
        const maxStart =
          (Number(first.end_sec) || dur) - MIN_SECTION_SEC;
        first.start_sec = Math.max(0, Math.min(maxStart, t));
      } else {
        const last = sections[sections.length - 1];
        const minEnd =
          (Number(last.start_sec) || 0) + MIN_SECTION_SEC;
        last.end_sec = Math.max(minEnd, Math.min(dur, t));
      }
      renderWaveRegions(song);
      updateCommitButton();
    };
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }

  function updateSectionPlayUi() {
    document.querySelectorAll(".wave-region").forEach((region) => {
      const song = region.dataset.song;
      const index = Number(region.dataset.index);
      const playBtn = region.querySelector(".wave-play");
      const head = region.querySelector(".wave-section-playhead");
      const isActive =
        activeSectionPreview &&
        activeSectionPreview.song === song &&
        activeSectionPreview.index === index;
      const isPlaying = isActive && !sectionPreviewAudio.paused;
      region.classList.toggle("is-active", Boolean(isActive));
      if (playBtn) {
        playBtn.textContent = isPlaying ? "❚❚" : "▶";
        playBtn.title = isPlaying ? "Pause section" : "Play section";
      }
      if (!head) return;
      if (!isActive) {
        head.hidden = true;
        return;
      }
      const start = activeSectionPreview.start;
      const end = activeSectionPreview.end;
      const secDur = Math.max(0.05, end - start);
      const t = sectionPreviewAudio.currentTime || 0;
      const ratio = Math.max(0, Math.min(1, t / secDur));
      head.hidden = false;
      head.style.left = `${ratio * 100}%`;
    });
  }

  async function toggleSectionPreview(song, index) {
    if (!sessionId) return;
    const sections = sectionsForSong(song);
    const sec = sections[index];
    if (!sec) return;
    const start = Number(sec.start_sec) || 0;
    const end = Number(sec.end_sec) || start + 1;

    const same =
      activeSectionPreview &&
      activeSectionPreview.song === song &&
      activeSectionPreview.index === index &&
      Math.abs(activeSectionPreview.start - start) < 0.001 &&
      Math.abs(activeSectionPreview.end - end) < 0.001;

    if (same && sectionPreviewAudio.src) {
      if (!sectionPreviewAudio.paused) {
        sectionPreviewAudio.pause();
        updateSectionPlayUi();
        return;
      }
      try {
        await sectionPreviewAudio.play();
        updateSectionPlayUi();
      } catch (err) {
        showError(err instanceof Error ? err.message : "Section play failed.");
      }
      return;
    }

    try {
      const response = await fetch(
        `/api/mashup/sessions/${sessionId}/song/${song}/section-preview?start=${start}&end=${end}`
      );
      if (!response.ok) throw new Error("Section preview failed");
      const blob = await response.blob();
      if (sectionPreviewUrl) URL.revokeObjectURL(sectionPreviewUrl);
      sectionPreviewUrl = URL.createObjectURL(blob);
      activeSectionPreview = { song, index, start, end };
      sectionPreviewAudio.src = sectionPreviewUrl;
      sectionPreviewAudio.load();
      await sectionPreviewAudio.play();
      updateSectionPlayUi();
    } catch (err) {
      showError(err instanceof Error ? err.message : "Section preview failed.");
    }
  }

  async function decodePeaks(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error("Failed to load song audio");
    const buffer = await response.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await ctx.decodeAudioData(buffer.slice(0));
    const channel = audioBuffer.getChannelData(0);
    const buckets = Math.min(4000, Math.max(200, Math.floor(audioBuffer.duration * PX_PER_SEC)));
    const peaks = new Float32Array(buckets);
    const block = Math.floor(channel.length / buckets) || 1;
    for (let i = 0; i < buckets; i += 1) {
      let max = 0;
      const start = i * block;
      for (let j = 0; j < block && start + j < channel.length; j += 1) {
        const v = Math.abs(channel[start + j]);
        if (v > max) max = v;
      }
      peaks[i] = max;
    }
    try {
      ctx.close();
    } catch {
      /* ignore */
    }
    return { peaks, duration: audioBuffer.duration };
  }

  async function loadWaveforms() {
    if (!sessionId || !studioWaveforms) return;
    studioWaveforms.hidden = false;
    const titleA = document.getElementById("wave-title-a");
    const titleB = document.getElementById("wave-title-b");
    if (titleA) titleA.textContent = studioState.title_a || "Song A";
    if (titleB) titleB.textContent = studioState.title_b || "Song B";
    try {
      const [a, b] = await Promise.all([
        decodePeaks(`/api/mashup/sessions/${sessionId}/song/a/audio`),
        decodePeaks(`/api/mashup/sessions/${sessionId}/song/b/audio`),
      ]);
      wavePeaks.a = a.peaks;
      wavePeaks.aDur = a.duration;
      wavePeaks.b = b.peaks;
      wavePeaks.bDur = b.duration;
      drawWaveform("a");
      drawWaveform("b");
      renderWaveRegions("a");
      renderWaveRegions("b");
      wireWaveformScrollSync();
    } catch (err) {
      console.warn(err);
      // Still show regions without peaks.
      wavePeaks.aDur = Math.max(
        ...(studioState.sections_a || []).map((s) => Number(s.end_sec) || 0),
        1
      );
      wavePeaks.bDur = Math.max(
        ...(studioState.sections_b || []).map((s) => Number(s.end_sec) || 0),
        1
      );
      renderWaveRegions("a");
      renderWaveRegions("b");
      wireWaveformScrollSync();
    }
  }

  async function commitPendingSections() {
    if (!sessionId || !studioState) return;
    if (pendingSections.a === null && pendingSections.b === null) return;
    if (studioAudio.src && Number.isFinite(studioAudio.currentTime)) {
      playheadTimeSec = studioAudio.currentTime;
    }
    const resumeAfter = Boolean(studioAudio.src && !studioAudio.paused);
    studioAudio.pause();
    setPlayIcons(false);
    setPlayheadVisual(playheadTimeSec);
    try {
      const response = await fetch(
        `/api/mashup/sessions/${sessionId}/studio/commit-sections`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sections_a: pendingSections.a,
            sections_b: pendingSections.b,
          }),
        }
      );
      if (!response.ok) throw new Error("Commit sections failed");
      studioState = await response.json();
      pendingSections = { a: null, b: null };
      updateCommitButton();
      renderStudioGrid();
      renderWaveRegions("a");
      renderWaveRegions("b");
      markStudioDirty();
      const token = ++studioRerenderToken;
      setRenderingUi(true);
      studioBusy = true;
      try {
        await ensureStudioPreviewLoaded();
        if (token !== studioRerenderToken) return;
        const max = studioAudio.duration || 0;
        const seekTo = Math.max(0, Math.min(playheadTimeSec, max || playheadTimeSec));
        studioAudio.currentTime = seekTo;
        playheadTimeSec = seekTo;
        setPlayheadVisual(playheadTimeSec);
        if (resumeAfter) {
          await studioAudio.play();
          setPlayIcons(true);
        }
      } finally {
        if (token === studioRerenderToken) {
          setRenderingUi(false);
          studioBusy = false;
        }
      }
    } catch (err) {
      showError(err instanceof Error ? err.message : "Commit failed.");
      setRenderingUi(false);
      studioBusy = false;
    }
  }

  async function loadStudioState() {
    if (!sessionId) return;
    const response = await fetch(`/api/mashup/sessions/${sessionId}/studio`);
    if (!response.ok) throw new Error("Failed to load section editor");
    studioState = await response.json();
    pendingSections = { a: null, b: null };
    studioDirty = true;
    studioAudioReady = false;
    playheadTimeSec = 0;
    updateCommitButton();
    renderStudioGrid();
    await loadWaveforms();
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

  async function ensureStudioPreviewLoaded() {
    if (!sessionId) return;
    if (!studioDirty && studioAudioReady && studioAudio.src) return;
    if (studioPlay && !studioPlay.classList.contains("rendering")) {
      setRenderingUi(true);
    }
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
    await new Promise((resolve, reject) => {
      const onReady = () => {
        studioAudio.removeEventListener("loadedmetadata", onReady);
        studioAudio.removeEventListener("error", onErr);
        resolve();
      };
      const onErr = () => {
        studioAudio.removeEventListener("loadedmetadata", onReady);
        studioAudio.removeEventListener("error", onErr);
        reject(new Error("Failed to load preview audio"));
      };
      studioAudio.addEventListener("loadedmetadata", onReady);
      studioAudio.addEventListener("error", onErr);
    });
    studioDirty = false;
    studioAudioReady = true;
    if (studioDownload) {
      studioDownload.hidden = false;
      studioDownload.href = studioObjectUrl;
    }
  }

  async function playStudioFromPlayhead({ forceStart = false } = {}) {
    if (!sessionId || !studioPlay || studioBusy) return;
    if (!studioAudio.paused && studioAudio.src && !forceStart && !studioDirty) {
      playheadTimeSec = studioAudio.currentTime;
      studioAudio.pause();
      setPlayIcons(false);
      return;
    }
    studioBusy = true;
    setRenderingUi(studioDirty || !studioAudioReady);
    try {
      const seekTo = forceStart ? 0 : playheadTimeSec;
      if (forceStart) playheadTimeSec = 0;
      await ensureStudioPreviewLoaded();
      const max = studioAudio.duration || 0;
      studioAudio.currentTime = Math.max(0, Math.min(seekTo, max || seekTo));
      playheadTimeSec = studioAudio.currentTime;
      setPlayheadVisual(playheadTimeSec);
      await studioAudio.play();
      setPlayIcons(true);
      if (studioPlayhead) studioPlayhead.hidden = false;
    } catch (err) {
      showError(err instanceof Error ? err.message : "Studio preview failed.");
      setPlayIcons(false);
    } finally {
      setRenderingUi(false);
      studioBusy = false;
    }
  }

  function updateStudioPlayhead() {
    if (!studioState) return;
    playheadTimeSec = studioAudio.currentTime || playheadTimeSec;
    setPlayheadVisual(playheadTimeSec);
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
    studioPlay.addEventListener("click", () => playStudioFromPlayhead());
  }
  if (studioSkip) {
    studioSkip.addEventListener("click", () =>
      playStudioFromPlayhead({ forceStart: true })
    );
  }
  if (commitSectionsBtn) {
    commitSectionsBtn.addEventListener("click", commitPendingSections);
  }
  studioAudio.addEventListener("timeupdate", updateStudioPlayhead);
  studioAudio.addEventListener("ended", () => {
    setPlayIcons(false);
    playheadTimeSec = studioAudio.duration || playheadTimeSec;
  });
  studioAudio.addEventListener("pause", () => setPlayIcons(false));
  studioAudio.addEventListener("play", () => setPlayIcons(true));

  if (sectionPreviewAudio) {
    sectionPreviewAudio.addEventListener("timeupdate", updateSectionPlayUi);
    sectionPreviewAudio.addEventListener("play", updateSectionPlayUi);
    sectionPreviewAudio.addEventListener("pause", updateSectionPlayUi);
    sectionPreviewAudio.addEventListener("ended", () => {
      if (sectionPreviewAudio) sectionPreviewAudio.currentTime = 0;
      updateSectionPlayUi();
    });
  }

  if (studioDownload) {
    studioDownload.addEventListener("click", async (event) => {
      if (!sessionId) return;
      if (studioDownload.dataset.ready === "1") {
        studioDownload.dataset.ready = "";
        return;
      }
      event.preventDefault();
      try {
        if (studioDirty || !studioObjectUrl) {
          await ensureStudioPreviewLoaded();
        }
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
