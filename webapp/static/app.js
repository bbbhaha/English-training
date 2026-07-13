const recordBtn = document.getElementById("recordBtn");
const recordText = document.getElementById("recordText");
const recordIcon = document.getElementById("recordIcon");
const playBtn = document.getElementById("playBtn");
const submitBtn = document.getElementById("submitBtn");
const audioFile = document.getElementById("audioFile");
const audioPreview = document.getElementById("audioPreview");
const targetText = document.getElementById("targetText");
const trimSilence = document.getElementById("trimSilence");
const resultBody = document.getElementById("resultBody");
const phoneTimeline = document.getElementById("phoneTimeline");
const phoneCount = document.getElementById("phoneCount");
const accentCount = document.getElementById("accentCount");
const errorCount = document.getElementById("errorCount");
const reviewCount = document.getElementById("reviewCount");
const modelStatus = document.getElementById("modelStatus");
const artifactRow = document.getElementById("artifactRow");
const debugPanel = document.getElementById("debugPanel");
const waveCanvas = document.getElementById("waveCanvas");
const canvasContext = waveCanvas.getContext("2d");

let mediaStream = null;
let audioContext = null;
let processor = null;
let sourceNode = null;
let recordedChunks = [];
let recordedBlob = null;
let isRecording = false;
let drawHandle = null;
let analyser = null;

function setStatus(text) {
  modelStatus.textContent = text;
}

function drawIdle() {
  const width = waveCanvas.width;
  const height = waveCanvas.height;
  canvasContext.clearRect(0, 0, width, height);
  canvasContext.fillStyle = "#14222d";
  canvasContext.fillRect(0, 0, width, height);
  canvasContext.strokeStyle = "#38bdf8";
  canvasContext.lineWidth = 2;
  canvasContext.beginPath();
  const mid = height / 2;
  for (let x = 0; x < width; x += 8) {
    const y = mid + Math.sin(x / 28) * 10 + Math.sin(x / 61) * 6;
    if (x === 0) canvasContext.moveTo(x, y);
    else canvasContext.lineTo(x, y);
  }
  canvasContext.stroke();
}

function drawLive() {
  if (!analyser) return;
  const data = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteTimeDomainData(data);
  canvasContext.fillStyle = "#14222d";
  canvasContext.fillRect(0, 0, waveCanvas.width, waveCanvas.height);
  canvasContext.strokeStyle = "#7dd3fc";
  canvasContext.lineWidth = 2;
  canvasContext.beginPath();
  const slice = waveCanvas.width / data.length;
  data.forEach((value, index) => {
    const x = index * slice;
    const y = (value / 255) * waveCanvas.height;
    if (index === 0) canvasContext.moveTo(x, y);
    else canvasContext.lineTo(x, y);
  });
  canvasContext.stroke();
  drawHandle = requestAnimationFrame(drawLive);
}

function mergeBuffers(chunks) {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(length);
  let offset = 0;
  chunks.forEach((chunk) => {
    merged.set(chunk, offset);
    offset += chunk.length;
  });
  return merged;
}

function writeString(view, offset, value) {
  for (let i = 0; i < value.length; i += 1) {
    view.setUint8(offset + i, value.charCodeAt(i));
  }
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }
  return new Blob([view], { type: "audio/wav" });
}

async function startRecording() {
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioContext = new AudioContext();
  recordedChunks = [];
  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 1024;
  processor = audioContext.createScriptProcessor(4096, 1, 1);
  processor.onaudioprocess = (event) => {
    recordedChunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
  };
  sourceNode.connect(analyser);
  sourceNode.connect(processor);
  processor.connect(audioContext.destination);
  isRecording = true;
  recordBtn.classList.add("recording");
  recordIcon.textContent = "■";
  recordText.textContent = "停止";
  setStatus("录音中");
  drawLive();
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  if (drawHandle) cancelAnimationFrame(drawHandle);
  processor?.disconnect();
  sourceNode?.disconnect();
  mediaStream?.getTracks().forEach((track) => track.stop());
  const samples = mergeBuffers(recordedChunks);
  recordedBlob = encodeWav(samples, audioContext.sampleRate);
  audioContext.close();
  audioPreview.src = URL.createObjectURL(recordedBlob);
  recordBtn.classList.remove("recording");
  recordIcon.textContent = "●";
  recordText.textContent = "录音";
  playBtn.disabled = false;
  submitBtn.disabled = false;
  setStatus("已录音");
  drawIdle();
}

recordBtn.addEventListener("click", async () => {
  try {
    if (isRecording) stopRecording();
    else await startRecording();
  } catch (error) {
    setStatus("麦克风不可用");
    alert(error.message);
  }
});

playBtn.addEventListener("click", () => {
  audioPreview.play();
});

audioFile.addEventListener("change", () => {
  const file = audioFile.files[0];
  if (!file) return;
  recordedBlob = file;
  audioPreview.src = URL.createObjectURL(file);
  playBtn.disabled = false;
  submitBtn.disabled = false;
  setStatus("已上传");
});

document.querySelectorAll(".prompt").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".prompt").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    targetText.value = button.dataset.text;
  });
});

function decisionText(row) {
  return row.display_decision || "正确";
}

function decisionClass(row) {
  if (row.display_error_type === "deletion") return "error";
  if (row.display_error_type === "possible_deletion" || row.display_error_type === "alignment_issue") return "review";
  return "correct";
}

function pct(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function errorDisplay(row) {
  return row.display_error || "0%";
}

function renderRows(rows) {
  phoneTimeline.innerHTML = "";
  resultBody.innerHTML = "";
  rows.forEach((row) => {
    const cls = decisionClass(row);
    const pill = document.createElement("div");
    pill.className = `phone-pill ${cls}`;
    pill.title = `${row.word} ${row.target_phone} ${decisionText(row)}`;
    pill.textContent = row.target_phone || "?";
    phoneTimeline.appendChild(pill);

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.word || ""}</td>
      <td><strong>${row.target_phone || ""}</strong></td>
      <td><span class="badge ${cls}">${row.display_decision || "正确"}</span></td>
      <td>${row.display_error || "0%"}</td>
      <td><span class="quality ${row.display_align === "bad" ? "bad" : "pass"}">${row.display_align || ""}</span></td>
      <td>${row.display_error_type || ""}</td>
      <td>${row.deletion_trigger_source || "none"}</td>
      <td class="reason">${row.missing_word_reason || ""}</td>
    `;
    resultBody.appendChild(tr);
  });
}

function renderDebug(debug) {
  debugPanel.textContent = debug ? JSON.stringify(debug, null, 2) : "";
}

function renderArtifacts(artifacts) {
  if (!artifacts) {
    artifactRow.innerHTML = "";
    return;
  }
  const items = [
    ["prediction", artifacts.prediction_csv],
    ["alignment", artifacts.alignment_csv],
    ["g2p", artifacts.g2p_json],
    ["word summary", artifacts.word_summary_csv],
    ["16k wav", artifacts.preprocessed_audio],
  ].filter((item) => item[1]);
  artifactRow.innerHTML = items.map(([label, value]) => `<span><b>${label}</b>${value}</span>`).join("");
}

submitBtn.addEventListener("click", async () => {
  if (!recordedBlob) return;
  const text = targetText.value.trim();
  if (!text) {
    alert("请输入目标文本");
    return;
  }
  setStatus("诊断中");
  submitBtn.disabled = true;
  const form = new FormData();
  form.append("text", text);
  form.append("utterance_id", `web_${Date.now()}`);
  form.append("speaker_id", "web_user");
  form.append("trim_silence", trimSilence.checked ? "1" : "0");
  form.append("audio", recordedBlob, "recording.wav");

  try {
    const response = await fetch("/api/diagnose", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "诊断失败");
    phoneCount.textContent = data.n_phones;
    accentCount.textContent = data.n_acceptable_accent;
    errorCount.textContent = data.n_true_error;
    reviewCount.textContent = data.n_uncertain_review;
    renderRows(data.rows);
    renderArtifacts(data.artifacts);
    renderDebug(data.debug);
    setStatus("完成");
  } catch (error) {
    setStatus("接口错误");
    alert(error.message);
  } finally {
    submitBtn.disabled = false;
  }
});

drawIdle();
