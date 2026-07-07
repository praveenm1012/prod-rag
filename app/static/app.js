const API_BASE = "/api/v1";

const healthCard = document.querySelector("#health-card .status-value");
const healthJson = document.querySelector("#health-json");
const providerCard = document.querySelector("#provider-card .status-value");
const providerJson = document.querySelector("#provider-json");
const uploadForm = document.querySelector("#upload-form");
const uploadFile = document.querySelector("#upload-file");
const uploadLabel = document.querySelector("#upload-label");
const uploadResult = document.querySelector("#upload-result");
const uploadProgress = document.querySelector("#upload-progress");
const uploadProgressTitle = document.querySelector("#upload-progress-title");
const uploadProgressElapsed = document.querySelector("#upload-progress-elapsed");
const uploadProgressBar = document.querySelector("#upload-progress-bar");
const uploadProgressDetail = document.querySelector("#upload-progress-detail");
const uploadSubmitBtn = document.querySelector("#upload-submit-btn");
const documentsList = document.querySelector("#documents-list");
const documentsEmpty = document.querySelector("#documents-empty");
const refreshDocumentsBtn = document.querySelector("#refresh-documents-btn");
const refreshStatusBtn = document.querySelector("#refresh-status-btn");
const chatForm = document.querySelector("#chat-form");
const chatQuestion = document.querySelector("#chat-question");
const chatUseRag = document.querySelector("#chat-use-rag");
const chatStream = document.querySelector("#chat-stream");
const chatTopK = document.querySelector("#chat-top-k");
const chatSubmitBtn = document.querySelector("#chat-submit-btn");
const chatEndpointBadge = document.querySelector("#chat-endpoint-badge");
const chatAnswer = document.querySelector("#chat-answer");
const chatMeta = document.querySelector("#chat-meta");
const chatCitations = document.querySelector("#chat-citations");
const chatResult = document.querySelector("#chat-result");
const requestLog = document.querySelector("#request-log");
const clearLogBtn = document.querySelector("#clear-log-btn");
const documentRowTemplate = document.querySelector("#document-row-template");

function prettyJson(value) {
  return JSON.stringify(value, null, 2);
}

function logRequest(method, path, status, detail = "") {
  const entry = document.createElement("div");
  entry.className = "log-entry";
  const timestamp = new Date().toLocaleTimeString();
  entry.innerHTML = `
    <time>${timestamp}</time>
    <div><strong>${method}</strong> ${path} → <strong>${status}</strong></div>
    ${detail ? `<div>${detail}</div>` : ""}
  `;
  requestLog.prepend(entry);
}

function setStatus(element, state, label) {
  element.dataset.state = state;
  element.textContent = label;
}

async function fetchJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const text = await response.text();
  let body = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
  }
  return { response, body };
}

async function refreshHealth() {
  setStatus(healthCard, "loading", "Checking…");
  try {
    const { response, body } = await fetchJson("/health");
    logRequest("GET", "/health", response.status);
    if (!response.ok) {
      throw new Error(typeof body === "string" ? body : body?.detail || "Health failed");
    }
    setStatus(healthCard, "ok", body.status.toUpperCase());
    healthJson.textContent = prettyJson(body);
  } catch (error) {
    setStatus(healthCard, "error", "Unavailable");
    healthJson.textContent = String(error);
  }
}

async function refreshProviderStatus() {
  setStatus(providerCard, "loading", "Checking…");
  try {
    const { response, body } = await fetchJson("/chat/status");
    logRequest("GET", "/chat/status", response.status);
    if (!response.ok) {
      throw new Error(body?.detail || "Provider status failed");
    }
    const label = body.configured ? `${body.provider} · ${body.model}` : "Not configured";
    setStatus(providerCard, body.configured ? "ok" : "warn", label);
    providerJson.textContent = prettyJson(body);
  } catch (error) {
    setStatus(providerCard, "error", "Unavailable");
    providerJson.textContent = String(error);
  }
}

async function refreshStatus() {
  await Promise.all([refreshHealth(), refreshProviderStatus()]);
}

function formatFileSize(bytes) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatPhase(phase) {
  const labels = {
    queued: "Queued",
    ingesting: "Reading file",
    chunking: "Splitting into chunks",
    embedding: "Generating embeddings (slowest step on CPU)",
    indexing: "Building search index",
  };
  return labels[phase] || phase || "Processing";
}

function setUploadProgress(visible, title = "", detail = "", percent = 8) {
  uploadProgress.hidden = !visible;
  uploadProgressTitle.textContent = title;
  uploadProgressDetail.textContent = detail;
  uploadProgressBar.style.width = `${percent}%`;
}

async function pollUploadStatus(documentId, startedAt) {
  const pollIntervalMs = 2000;
  const maxAttempts = 900;

  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const elapsedSeconds = Math.floor((Date.now() - startedAt) / 1000);
    uploadProgressElapsed.textContent = `${elapsedSeconds}s`;

    const { response, body } = await fetchJson(`/upload/${documentId}/status`);
    if (!response.ok) {
      throw new Error(body?.detail || "Failed to fetch upload status");
    }

    const phasePercent = {
      queued: 12,
      ingesting: 24,
      chunking: 38,
      embedding: 72,
      indexing: 92,
    };
    const percent = body.status === "indexed" ? 100 : phasePercent[body.phase] || 15;
    setUploadProgress(
      true,
      body.status === "indexed" ? "Indexing complete" : "Processing upload…",
      `${formatPhase(body.phase)} · ${body.chunk_count} chunks · ${formatFileSize(body.file_size_bytes)}`,
      percent,
    );

    if (body.status === "indexed") {
      return body;
    }
    if (body.status === "failed") {
      throw new Error(body.error_message || "Upload failed during background processing");
    }

    await refreshDocuments();
    await new Promise((resolve) => window.setTimeout(resolve, pollIntervalMs));
  }

  throw new Error("Upload is still processing. Check the documents list and try again later.");
}

function renderDocuments(payload) {
  documentsList.innerHTML = "";
  const documents = payload?.documents || [];
  documentsEmpty.hidden = documents.length > 0;

  for (const document of documents) {
    const row = documentRowTemplate.content.firstElementChild.cloneNode(true);
    const statusClass = document.status || "indexed";
    row.querySelector(".document-name").innerHTML =
      `${escapeHtml(document.filename)}` +
      `<span class="document-status ${statusClass}">${statusClass}</span>`;
    const phaseSuffix = document.phase ? ` · ${formatPhase(document.phase)}` : "";
    row.querySelector(".document-meta").textContent =
      `${document.chunk_count} chunks · ${formatFileSize(document.file_size_bytes || 0)}` +
      `${phaseSuffix} · ${document.document_id}`;
    const deleteButton = row.querySelector(".delete-btn");
    const askButton = row.querySelector(".ask-btn");
    if (document.status === "processing") {
      deleteButton.textContent = "Cancel";
      askButton.disabled = true;
    }
    askButton.addEventListener("click", () => {
      chatQuestion.value = `Tell me about ${document.filename}`;
      chatQuestion.focus();
    });
    deleteButton.addEventListener("click", async () => {
      await deleteDocument(document.document_id, document.filename);
    });
    documentsList.appendChild(row);
  }
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

async function refreshDocuments() {
  try {
    const { response, body } = await fetchJson("/documents");
    logRequest("GET", "/documents", response.status, `${body.total} document(s)`);
    if (!response.ok) {
      throw new Error(body?.detail || "Failed to list documents");
    }
    renderDocuments(body);
  } catch (error) {
    documentsEmpty.hidden = false;
    documentsEmpty.textContent = String(error);
  }
}

async function deleteDocument(documentId, filename) {
  if (!window.confirm(`Delete ${filename}?`)) {
    return;
  }
  const response = await fetch(`${API_BASE}/delete`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_id: documentId }),
  });
  logRequest("DELETE", "/delete", response.status, documentId);
  if (!response.ok) {
    const body = await response.json();
    window.alert(body.detail || "Delete failed");
    return;
  }
  await refreshDocuments();
}

uploadFile.addEventListener("change", () => {
  const file = uploadFile.files[0];
  if (!file) {
    uploadLabel.textContent = "Choose PDF, TXT, Markdown, or DOCX";
    return;
  }
  uploadLabel.textContent = `${file.name} (${formatFileSize(file.size)})`;
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = uploadFile.files[0];
  if (!file) {
    window.alert("Choose a file to upload.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  uploadResult.textContent = "Upload accepted. Processing in background…";
  uploadSubmitBtn.disabled = true;
  const startedAt = Date.now();
  setUploadProgress(
    true,
    "Uploading file…",
    `${file.name} · ${formatFileSize(file.size)}`,
    5,
  );

  try {
    const response = await fetch(`${API_BASE}/upload?background=true`, {
      method: "POST",
      body: formData,
    });
    const body = await response.json();
    logRequest("POST", "/upload?background=true", response.status, file.name);
    uploadResult.textContent = prettyJson(body);
    if (response.status !== 202) {
      throw new Error(body.detail || `Upload failed with status ${response.status}`);
    }

    const documentId = body.document.document_id;
    const finalStatus = await pollUploadStatus(documentId, startedAt);
    uploadResult.textContent = prettyJson({
      ...body,
      final_status: finalStatus,
    });
    uploadFile.value = "";
    uploadLabel.textContent = "Choose PDF, TXT, Markdown, or DOCX";
    setUploadProgress(
      true,
      "Indexing complete",
      `${finalStatus.chunk_count} chunks indexed`,
      100,
    );
    await refreshDocuments();
  } catch (error) {
    uploadResult.textContent = String(error);
    setUploadProgress(true, "Upload failed", String(error), 100);
  } finally {
    uploadSubmitBtn.disabled = false;
  }
});

function renderChatResponse(body, streamedText = "") {
  chatAnswer.textContent = streamedText || body.answer || "No response.";
  chatMeta.textContent = body.provider
    ? `${body.provider} · ${body.model}`
    : streamedText
      ? "Streaming response"
      : "";
  chatCitations.innerHTML = "";
  for (const citation of body.citations || []) {
    const pill = document.createElement("span");
    pill.className = "citation-pill";
    pill.textContent = citation;
    chatCitations.appendChild(pill);
  }
  chatResult.textContent = prettyJson(body);
}

async function sendChat(event) {
  event.preventDefault();
  const question = chatQuestion.value.trim();
  if (!question) {
    return;
  }

  const payload = {
    question,
    use_rag: chatUseRag.checked,
    top_k: Number(chatTopK.value) || 3,
  };

  chatSubmitBtn.disabled = true;
  chatAnswer.textContent = "Waiting for response…";
  chatMeta.textContent = "";
  chatCitations.innerHTML = "";
  chatResult.textContent = prettyJson(payload);

  try {
    if (chatStream.checked) {
      chatEndpointBadge.textContent = "POST /api/v1/chat/stream";
      await streamChat(payload);
    } else {
      chatEndpointBadge.textContent = "POST /api/v1/chat";
      const { response, body } = await fetchJson("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      logRequest("POST", "/chat", response.status);
      if (!response.ok) {
        throw new Error(body?.detail || "Chat failed");
      }
      renderChatResponse(body);
    }
  } catch (error) {
    chatAnswer.textContent = String(error);
    chatResult.textContent = String(error);
  } finally {
    chatSubmitBtn.disabled = false;
  }
}

async function streamChat(payload) {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  logRequest("POST", "/chat/stream", response.status);
  if (!response.ok) {
    const body = await response.json();
    throw new Error(body.detail || "Streaming chat failed");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let streamedText = "";
  chatAnswer.textContent = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    streamedText += decoder.decode(value, { stream: true });
    chatAnswer.textContent = streamedText;
  }

  renderChatResponse({ answer: streamedText, citations: [] }, streamedText);
}

chatForm.addEventListener("submit", sendChat);
refreshDocumentsBtn.addEventListener("click", refreshDocuments);
refreshStatusBtn.addEventListener("click", refreshStatus);
clearLogBtn.addEventListener("click", () => {
  requestLog.innerHTML = "";
});

refreshStatus();
refreshDocuments();
