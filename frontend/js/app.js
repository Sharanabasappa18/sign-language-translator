const video = document.getElementById("video");
const annotated = document.getElementById("annotated");
const overlay = document.getElementById("overlay");
const placeholder = document.getElementById("cameraPlaceholder");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const clearBtn = document.getElementById("clearBtn");
const speakBtn = document.getElementById("speakBtn");
const languageSelect = document.getElementById("languageSelect");
const ttsToggle = document.getElementById("ttsToggle");
const cameraStatus = document.getElementById("cameraStatus");
const modelBadge = document.getElementById("modelBadge");
const gestureLabel = document.getElementById("gestureLabel");
const confidenceLabel = document.getElementById("confidenceLabel");
const previewText = document.getElementById("previewText");
const translatedText = document.getElementById("translatedText");
const gestureList = document.getElementById("gestureList");

let stream = null;
let ws = null;
let sending = false;
let frameTimer = null;

const TTS_CODES = {
  en: "en-US",
  kn: "kn-IN",
  hi: "hi-IN",
  ta: "ta-IN",
};

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/ws/recognize`;
}

async function init() {
  const [health, gestures] = await Promise.all([
    fetch("/api/health").then((r) => r.json()),
    fetch("/api/gestures").then((r) => r.json()),
  ]);

  modelBadge.textContent = health.model_loaded ? "ML model loaded" : "Rule-based mode";
  modelBadge.className = health.model_loaded ? "badge badge-on" : "badge badge-muted";

  gestureList.innerHTML = gestures.gestures
    .map((g) => `<li>${g.replace(/_/g, " ")}</li>`)
    .join("");
}

function connectWebSocket() {
  ws = new WebSocket(wsUrl());

  ws.onopen = () => {
    sendConfig();
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "result") {
      handleResult(data);
    } else if (data.type === "clear" || data.type === "reset") {
      translatedText.textContent = "";
      speakBtn.disabled = true;
    }
  };

  ws.onclose = () => {
    if (stream) {
      setTimeout(connectWebSocket, 1000);
    }
  };
}

function sendConfig() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "config", language: languageSelect.value }));
  }
}

function handleResult(data) {
  if (data.annotated_frame) {
    annotated.src = `data:image/jpeg;base64,${data.annotated_frame}`;
    annotated.hidden = false;
    video.hidden = true;
  }

  gestureLabel.textContent = data.current_gesture
    ? data.current_gesture.replace(/_/g, " ")
    : "—";
  confidenceLabel.textContent = data.confidence
    ? `${Math.round(data.confidence * 100)}%`
    : "—";
  previewText.textContent = data.preview_text || "Hold a sign steady to translate";

  if (data.translated_text !== undefined) {
    translatedText.textContent = data.translated_text;
    speakBtn.disabled = !data.translated_text.trim();
  }

  if (data.accepted_gesture) {
    previewText.classList.add("flash");
    setTimeout(() => previewText.classList.remove("flash"), 450);

    if (ttsToggle.checked && data.preview_text) {
      speak(data.preview_text);
    }
  }
}

function captureAndSend() {
  if (!ws || ws.readyState !== WebSocket.OPEN || sending) return;
  sending = true;

  const ctx = overlay.getContext("2d");
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);

  overlay.toBlob(
    (blob) => {
      if (!blob) {
        sending = false;
        return;
      }
      const reader = new FileReader();
      reader.onloadend = () => {
        ws.send(JSON.stringify({ type: "frame", data: reader.result }));
        sending = false;
      };
      reader.readAsDataURL(blob);
    },
    "image/jpeg",
    0.75
  );
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: 640, height: 480 },
      audio: false,
    });
    video.srcObject = stream;
    video.hidden = false;
    annotated.hidden = true;
    placeholder.classList.add("hidden");

    connectWebSocket();
    frameTimer = setInterval(captureAndSend, 120);

    cameraStatus.textContent = "Live";
    cameraStatus.className = "badge badge-on";
    startBtn.disabled = true;
    stopBtn.disabled = false;
  } catch (err) {
    alert("Could not access webcam. Please allow camera permissions.");
    console.error(err);
  }
}

function stopCamera() {
  if (frameTimer) {
    clearInterval(frameTimer);
    frameTimer = null;
  }
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }

  video.srcObject = null;
  video.hidden = false;
  annotated.hidden = true;
  placeholder.classList.remove("hidden");

  cameraStatus.textContent = "Stopped";
  cameraStatus.className = "badge badge-off";
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

function speak(text) {
  if (!text) return;

  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = TTS_CODES[languageSelect.value] || "en-US";
    utterance.rate = 0.95;
    window.speechSynthesis.speak(utterance);
    return;
  }

  fetch("/api/tts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, language: languageSelect.value }),
  })
    .then((r) => r.blob())
    .then((blob) => {
      const audio = new Audio(URL.createObjectURL(blob));
      audio.play();
    })
    .catch(console.error);
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);

clearBtn.addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "clear" }));
  }
  translatedText.textContent = "";
  speakBtn.disabled = true;
});

speakBtn.addEventListener("click", () => {
  speak(translatedText.textContent.trim());
});

languageSelect.addEventListener("change", sendConfig);

init();
