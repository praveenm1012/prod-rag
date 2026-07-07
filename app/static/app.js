const API_BASE = "/api/v1";

const healthCard = document.querySelector("#health-card .status-value");
const healthJson = document.querySelector("#health-json");
const providerCard = document.querySelector("#provider-card .status-value");
const providerJson = document.querySelector("#provider-json");
const uploadForm = document.querySelector("#upload-form");
const uploadFile = document.querySelector("#upload-file");
const uploadLabel = document.querySelector("#upload-label");
const uploadResult = document.querySelector("#upload-result");
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

function renderDocuments(payload) {
  documentsList.innerHTML = "";
  const documents = payload?.documents || [];
  documentsEmpty.hidden = documents.length > 0;

  for (const document of documents) {
    const row = documentRowTemplate.content.firstElementChild.cloneNode(true);
    row.querySelector(".document-name").textContent = document.filename;
    row.querySelector(".document-meta").textContent =
      `${document.chunk_count} chunks · ${document.document_id}`;
    row.querySelector(".ask-btn").addEventListener("click", () => {
      chatQuestion.value = `Tell me about ${document.filename}`;
      chatQuestion.focus();
    });
    row.querySelector(".delete-btn").addEventListener("click", async () => {
      await deleteDocument(document.document_id, document.filename);
    });
    documentsList.appendChild(row);
  }
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
  uploadLabel.textContent = uploadFile.files[0]?.name || "Choose PDF, TXT, Markdown, or DOCX";
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
  uploadResult.textContent = "Uploading and indexing…";

  try {
    const response = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData,
    });
    const body = await response.json();
    logRequest("POST", "/upload", response.status, file.name);
    uploadResult.textContent = prettyJson(body);
    if (!response.ok) {
      throw new Error(body.detail || "Upload failed");
    }
    uploadFile.value = "";
    uploadLabel.textContent = "Choose PDF, TXT, Markdown, or DOCX";
    await refreshDocuments();
  } catch (error) {
    uploadResult.textContent = String(error);
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
