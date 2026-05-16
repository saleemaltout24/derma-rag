import React, { useEffect, useMemo, useRef, useState } from "react";
import { Moon, Sun, Upload, Send, RefreshCcw, Image as ImageIcon } from "lucide-react";

const ACCEPTED_IMAGE_TYPES = ".jpg,.jpeg,.png,.webp";

function pickClassification(data) {
  return data?.classification || data?.classifier_result || null;
}

function isClassifierUnavailable(result) {
  if (!result) return false;
  const code = result.predicted_class;
  return code === "UNAVAILABLE" || code === "UNKNOWN" || (result.confidence ?? 0) === 0;
}

function formatTextbookMatchLabel(match, index) {
  const book = match.source_pdf || match.source;
  const page = match.page;
  const file = match.file_name;
  const parts = [];
  if (book) parts.push(String(book));
  if (page != null && page !== "") parts.push(`page ${page}`);
  if (file) parts.push(String(file));
  if (parts.length) return parts.join(" · ");
  return match.disease || match.caption || `Match ${index + 1}`;
}

async function parseApiResponse(response) {
  let data = {};
  try {
    data = await response.json();
  } catch {
    data = {};
  }
  if (!response.ok) {
    const detail = data.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
          : `Request failed (${response.status})`;
    throw new Error(message);
  }
  return data;
}

export default function App() {
  const [darkMode, setDarkMode] = useState(true);
  const [apiBase, setApiBase] = useState("http://127.0.0.1:8000");
  const [sessionId, setSessionId] = useState("default");
  const [question, setQuestion] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hi — I'm your dermatology assistant. You can ask text questions, upload a skin image, or do both together.",
    },
  ]);
  const [structuredState, setStructuredState] = useState(null);
  const [lastResponse, setLastResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [classification, setClassification] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  const theme = useMemo(() => ({
    page: darkMode ? "bg-zinc-950 text-zinc-100" : "bg-zinc-100 text-zinc-900",
    card: darkMode ? "bg-zinc-900 border-zinc-800" : "bg-white border-zinc-200",
    soft: darkMode ? "bg-zinc-900/60" : "bg-zinc-50",
    input: darkMode ? "bg-zinc-950 border-zinc-800 text-zinc-100" : "bg-white border-zinc-300 text-zinc-900",
    bubbleUser: "bg-blue-600 text-white",
    bubbleAssistant: darkMode ? "bg-zinc-800 text-zinc-100" : "bg-zinc-200 text-zinc-900",
    muted: darkMode ? "text-zinc-400" : "text-zinc-500",
  }), [darkMode]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleFileChange = (event) => {
    const file = event.target.files?.[0] || null;
    setSelectedFile(file);
    setClassification(null);
    setHeatmapUrl(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (file) {
      setPreviewUrl(URL.createObjectURL(file));
    } else {
      setPreviewUrl("");
    }
  };

  const sendMessage = async () => {
    if (!question.trim() && !selectedFile) return;
    setLoading(true);
    setHeatmapUrl(null);

    try {
      const formData = new FormData();
      formData.append("session_id", sessionId);
      formData.append("question", question);
      if (selectedFile) formData.append("file", selectedFile);

      setMessages((prev) => [
        ...prev,
        { role: "user", content: question.trim() || "[uploaded image]" },
      ]);

      const response = await fetch(`${apiBase}/chat`, {
        method: "POST",
        body: formData,
      });

      const data = await parseApiResponse(response);
      setLastResponse(data);
      setStructuredState(data.structured_state ?? null);

      const cls = pickClassification(data);
      setClassification(cls);
      if (data.heatmap) setHeatmapUrl(data.heatmap);

      const answer = data.answer || "No response received.";
      setMessages((prev) => [...prev, { role: "assistant", content: answer }]);

      setQuestion("");
      setSelectedFile(null);
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl("");
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `❌ ${error.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const resetChat = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${apiBase}/reset?session_id=${encodeURIComponent(sessionId)}`, {
        method: "POST",
      });
      const data = await parseApiResponse(response);
      setMessages([{ role: "assistant", content: data.message || "Chat reset." }]);
      setStructuredState(null);
      setLastResponse(null);
      setClassification(null);
      setHeatmapUrl(null);
      setQuestion("");
      setSelectedFile(null);
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl("");
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (error) {
      setMessages((prev) => [...prev, { role: "assistant", content: `Reset failed: ${error.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const imageMatches = lastResponse?.image_matches || [];
  const showDiagnosis = classification && !isClassifierUnavailable(classification);

  return (
    <div className={`min-h-screen ${theme.page} transition-colors`}>
      <div className="mx-auto max-w-7xl px-4 py-6 md:px-6 lg:px-8">
        <div className="mb-6 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight">🩺 Derma RAG Assistant</h1>
            <p className={`mt-1 text-sm ${theme.muted}`}>
              AI-powered dermatology assistant with deep learning diagnosis
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button onClick={() => setDarkMode((v) => !v)} className={`rounded-2xl border px-4 py-2 ${theme.input}`}>
              {darkMode ? <Sun className="inline h-4 w-4 mr-2" /> : <Moon className="inline h-4 w-4 mr-2" />}
              {darkMode ? "Light" : "Dark"}
            </button>
            <button onClick={resetChat} className="rounded-2xl bg-blue-600 px-4 py-2 text-white">
              <RefreshCcw className="inline h-4 w-4 mr-2" />Reset
            </button>
          </div>
        </div>

        <div className="grid gap-6 xl:grid-cols-[minmax(680px,1.2fr)_minmax(360px,0.8fr)]">
          <div className={`min-w-0 rounded-3xl border p-5 shadow-xl ${theme.card}`}>
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold">Chat</h2>
              <span className="rounded-full bg-zinc-700 px-3 py-1 text-xs text-white">Session: {sessionId}</span>
            </div>

            <div className={`mb-4 h-[520px] overflow-y-auto rounded-2xl border p-4 ${theme.soft}`}>
              <div className="space-y-4">
                {messages.map((message, index) => (
                  <div key={index} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                    <div
                      className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-6 shadow-sm whitespace-pre-wrap ${
                        message.role === "user" ? theme.bubbleUser : theme.bubbleAssistant
                      }`}
                    >
                      {message.content}
                    </div>
                  </div>
                ))}
                {loading && (
                  <div className="flex justify-start">
                    <div className={`rounded-2xl px-4 py-3 text-sm ${theme.bubbleAssistant}`}>
                      ⏳ Thinking...
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            </div>

            <div className="space-y-4">
              <div className="grid gap-3 md:grid-cols-2">
                <input
                  value={apiBase}
                  onChange={(e) => setApiBase(e.target.value)}
                  placeholder="API base URL"
                  className={`rounded-2xl border px-4 py-3 ${theme.input}`}
                />
                <input
                  value={sessionId}
                  onChange={(e) => setSessionId(e.target.value)}
                  placeholder="Session ID"
                  className={`rounded-2xl border px-4 py-3 ${theme.input}`}
                />
              </div>

              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                  }
                }}
                placeholder="Type a dermatology question... (Enter to send)"
                className={`min-h-[100px] w-full rounded-2xl border px-4 py-3 ${theme.input}`}
              />

              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div className="flex flex-wrap items-center gap-3">
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className={`rounded-2xl border px-4 py-2 ${theme.input}`}
                  >
                    <Upload className="inline h-4 w-4 mr-2" />Upload Image
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept={ACCEPTED_IMAGE_TYPES}
                    onChange={handleFileChange}
                    className="hidden"
                  />
                  {selectedFile && (
                    <span className="rounded-full bg-zinc-700 px-3 py-1 text-xs text-white">{selectedFile.name}</span>
                  )}
                </div>
                <button onClick={sendMessage} disabled={loading} className="rounded-2xl bg-blue-600 px-6 py-2 text-white disabled:opacity-50">
                  <Send className="inline h-4 w-4 mr-2" />Send
                </button>
              </div>

              {previewUrl && (
                <div className={`rounded-2xl border p-3 ${theme.soft}`}>
                  <div className="mb-2 flex items-center gap-2 text-sm font-medium">
                    <ImageIcon className="h-4 w-4" />Image Preview
                  </div>
                  <img src={previewUrl} alt="Preview" className="max-h-48 rounded-2xl object-contain" />
                </div>
              )}
            </div>
          </div>

          <div className="min-w-0 space-y-6">
            <div className={`rounded-3xl border p-5 shadow-xl ${theme.card}`}>
              <h2 className="mb-4 text-lg font-semibold">🔬 AI Diagnosis</h2>
              {showDiagnosis ? (
                <div className="space-y-3">
                  {(() => {
                    const tier =
                      classification.confidence_tier ??
                      (classification.confidence > 70
                        ? "high"
                        : classification.confidence >= 40
                          ? "medium"
                          : "low");
                    const barFillClass =
                      tier === "high" ? "bg-green-500" : tier === "medium" ? "bg-yellow-500" : "bg-red-500";
                    const showLowWarning = classification.ambiguous === true || tier === "low";
                    const showMediumInfo = !showLowWarning && tier === "medium";
                    const runnerUp =
                      classification.top2_name != null && classification.top2_name !== "";

                    return (
                      <>
                        <div
                          className={`rounded-2xl p-4 ${
                            darkMode ? "bg-blue-900/40 border border-blue-700" : "bg-blue-50 border border-blue-200"
                          }`}
                        >
                          <div className="text-xs text-blue-400 mb-1">Most Likely Condition</div>
                          <div className="text-lg font-bold">{classification.predicted_name}</div>
                          <div className="text-sm text-zinc-400">{classification.confidence.toFixed(1)}% confidence</div>
                          {runnerUp && (
                            <div
                              className={`mt-2 border-t pt-2 text-sm ${
                                darkMode ? "border-blue-800 text-zinc-300" : "border-blue-200 text-zinc-600"
                              }`}
                            >
                              <span className="font-medium">Runner-up:</span> {classification.top2_name}{" "}
                              <span className={theme.muted}>
                                ({Number(classification.top2_confidence).toFixed(1)}%)
                              </span>
                            </div>
                          )}
                        </div>
                        {showLowWarning && (
                          <div
                            className={`rounded-2xl border px-3 py-2 text-xs leading-relaxed ${
                              darkMode
                                ? "border-yellow-600/70 bg-yellow-950/50 text-yellow-100"
                                : "border-amber-400 bg-amber-50 text-amber-950"
                            }`}
                          >
                            ⚠️ Low confidence — model is uncertain. Results are approximate.
                          </div>
                        )}
                        {showMediumInfo && (
                          <div
                            className={`rounded-2xl border px-3 py-2 text-xs leading-relaxed ${
                              darkMode
                                ? "border-blue-500/60 bg-blue-950/40 text-blue-100"
                                : "border-blue-300 bg-blue-50 text-blue-950"
                            }`}
                          >
                            ℹ️ Moderate confidence — use as a guide only.
                          </div>
                        )}
                        <div className="space-y-2">
                          {classification.all_predictions?.map((pred) => (
                            <div key={pred.code} className="space-y-1">
                              <div className="flex justify-between text-xs">
                                <span>{pred.name}</span>
                                <span>{pred.confidence.toFixed(1)}%</span>
                              </div>
                              <div className="h-2 rounded-full bg-zinc-700">
                                <div
                                  className={`h-2 rounded-full ${barFillClass}`}
                                  style={{ width: `${pred.confidence}%` }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      </>
                    );
                  })()}
                </div>
              ) : classification && isClassifierUnavailable(classification) ? (
                <p className={`text-sm ${theme.muted}`}>
                  Classifier weights not loaded. Add <code className="text-xs">models/skin_classifier_v2.pth</code> and
                  restart the backend.
                </p>
              ) : (
                <p className={`text-sm ${theme.muted}`}>Upload a skin image to see AI diagnosis.</p>
              )}
            </div>

            <div className={`rounded-3xl border p-5 shadow-xl ${theme.card}`}>
              <h2 className="mb-4 text-lg font-semibold">🌡️ Grad-CAM Heatmap</h2>
              {loading && selectedFile ? (
                <p className={`text-sm ${theme.muted}`}>⏳ Generating heatmap...</p>
              ) : heatmapUrl ? (
                <div className="space-y-2">
                  <p className={`text-xs ${theme.muted}`}>Red areas = where the AI focused most (top predicted class)</p>
                  <img src={heatmapUrl} alt="Grad-CAM Heatmap" className="w-full rounded-2xl" />
                </div>
              ) : (
                <p className={`text-sm ${theme.muted}`}>Upload a skin image to see the heatmap.</p>
              )}
            </div>

            {imageMatches.length > 0 && (
              <div className={`rounded-3xl border p-5 shadow-xl ${theme.card}`}>
                <h2 className="mb-4 text-lg font-semibold">📚 Similar textbook images</h2>
                <ul className={`space-y-3 text-sm ${theme.muted}`}>
                  {imageMatches.slice(0, 5).map((match, i) => {
                    const label = formatTextbookMatchLabel(match, i);
                    const extra = [match.disease, match.caption].filter(Boolean).join(" — ");
                    return (
                      <li key={i} className="leading-snug">
                        <div className="truncate font-medium text-slate-800 dark:text-slate-100">
                          {label}
                          {match.score != null && (
                            <span className="ml-1 font-normal opacity-70">
                              ({Number(match.score).toFixed(2)})
                            </span>
                          )}
                        </div>
                        {extra && <div className="truncate text-xs opacity-80">{extra}</div>}
                        {match.body_site && (
                          <div className="truncate text-xs opacity-70">Site: {match.body_site}</div>
                        )}
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}

            <div className={`rounded-3xl border p-5 shadow-xl ${theme.card}`}>
              <h2 className="mb-4 text-lg font-semibold">Structured State</h2>
              <pre className={`overflow-auto whitespace-pre-wrap break-words rounded-2xl p-4 text-xs leading-6 ${theme.soft}`}>
                {structuredState != null
                  ? JSON.stringify(structuredState, null, 2)
                  : "No structured state yet."}
              </pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
