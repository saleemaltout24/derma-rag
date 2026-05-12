import React, { useMemo, useRef, useState } from "react";
import { Moon, Sun, Upload, Send, RefreshCcw, Image as ImageIcon } from "lucide-react";

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
      content:
        "Hi — I’m your dermatology assistant. You can ask text questions, upload a skin image, or do both together.",
    },
  ]);
  const [structuredState, setStructuredState] = useState(null);
  const [lastResponse, setLastResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);

  const theme = useMemo(
    () => ({
      page: darkMode ? "bg-zinc-950 text-zinc-100" : "bg-zinc-100 text-zinc-900",
      card: darkMode ? "bg-zinc-900 border-zinc-800" : "bg-white border-zinc-200",
      soft: darkMode ? "bg-zinc-900/60" : "bg-zinc-50",
      input: darkMode ? "bg-zinc-950 border-zinc-800 text-zinc-100" : "bg-white border-zinc-300 text-zinc-900",
      bubbleUser: "bg-blue-600 text-white",
      bubbleAssistant: darkMode ? "bg-zinc-800 text-zinc-100" : "bg-zinc-200 text-zinc-900",
      muted: darkMode ? "text-zinc-400" : "text-zinc-500",
    }),
    [darkMode]
  );

  const textMatches = lastResponse?.text_matches || [];
  const imageMatches = lastResponse?.image_matches || [];
  const classifierResult = lastResponse?.classifier_result || null;
  const retrievalDebug = lastResponse?.retrieval_debug || null;

  const shortText = (value, limit = 280) => {
    if (!value) return "";
    return value.length > limit ? `${value.slice(0, limit)}...` : value;
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0] || null;
    setSelectedFile(file);

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

    try {
      const formData = new FormData();
      formData.append("session_id", sessionId);
      formData.append("question", question);
      if (selectedFile) {
        formData.append("file", selectedFile);
      }

      setMessages((prev) => [
        ...prev,
        { role: "user", content: question.trim() || "[uploaded image]" },
      ]);

      const response = await fetch(`${apiBase}/chat`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        const detail = data?.detail || data?.error || "Request failed.";
        throw new Error(detail);
      }
      setLastResponse(data);
      setStructuredState(data.structured_state || null);

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
        { role: "assistant", content: `Request failed: ${error.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const resetChat = async () => {
    try {
      setLoading(true);
      const response = await fetch(
        `${apiBase}/reset?session_id=${encodeURIComponent(sessionId)}`,
        { method: "POST" }
      );
      const data = await response.json();
      if (!response.ok) {
        const detail = data?.detail || data?.error || "Reset failed.";
        throw new Error(detail);
      }

      setMessages([
        { role: "assistant", content: data.message || "Chat reset." },
      ]);
      setStructuredState(null);
      setLastResponse(null);
      setQuestion("");
      setSelectedFile(null);
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl("");
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Reset failed: ${error.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`min-h-screen ${theme.page} transition-colors`}>
      <div className="mx-auto max-w-7xl px-4 py-6 md:px-6 lg:px-8">
        <div className="mb-6 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight">Derma RAG Assistant</h1>
            <p className={`mt-1 text-sm ${theme.muted}`}>
              
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => setDarkMode((v) => !v)}
              className={`rounded-2xl border px-4 py-2 ${theme.input}`}
            >
              {darkMode ? <Sun className="inline h-4 w-4 mr-2" /> : <Moon className="inline h-4 w-4 mr-2" />}
              {darkMode ? "Light" : "Dark"}
            </button>
            <button
              onClick={resetChat}
              className="rounded-2xl bg-blue-600 px-4 py-2 text-white"
            >
              <RefreshCcw className="inline h-4 w-4 mr-2" />
              Reset
            </button>
          </div>
        </div>

        <div className="grid gap-6 xl:grid-cols-[minmax(680px,1.2fr)_minmax(360px,0.8fr)]">
          <div className={`min-w-0 rounded-3xl border p-5 shadow-xl ${theme.card}`}>
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold">Chat</h2>
              <span className="rounded-full bg-zinc-700 px-3 py-1 text-xs text-white">
                Session: {sessionId}
              </span>
            </div>

            <div className={`mb-4 h-[520px] overflow-y-auto rounded-2xl border p-4 ${theme.soft}`}>
              <div className="space-y-4">
                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-6 shadow-sm ${
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
                      Thinking...
                    </div>
                  </div>
                )}
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
                placeholder="Type a dermatology question..."
                className={`min-h-[120px] w-full rounded-2xl border px-4 py-3 ${theme.input}`}
              />

              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div className="flex flex-wrap items-center gap-3">
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className={`rounded-2xl border px-4 py-2 ${theme.input}`}
                  >
                    <Upload className="inline h-4 w-4 mr-2" />
                    Upload Image
                  </button>

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="hidden"
                  />

                  {selectedFile && (
                    <span className="rounded-full bg-zinc-700 px-3 py-1 text-xs text-white">
                      {selectedFile.name}
                    </span>
                  )}
                </div>

                <button
                  onClick={sendMessage}
                  disabled={loading}
                  className="rounded-2xl bg-blue-600 px-6 py-2 text-white"
                >
                  <Send className="inline h-4 w-4 mr-2" />
                  Send
                </button>
              </div>

              {previewUrl && (
                <div className={`rounded-2xl border p-3 ${theme.soft}`}>
                  <div className="mb-2 flex items-center gap-2 text-sm font-medium">
                    <ImageIcon className="h-4 w-4" />
                    Image Preview
                  </div>
                  <img
                    src={previewUrl}
                    alt="Preview"
                    className="max-h-64 rounded-2xl object-contain"
                  />
                </div>
              )}
            </div>
          </div>

          <div className="min-w-0 space-y-6">
            <div className={`rounded-3xl border p-5 shadow-xl ${theme.card}`}>
              <h2 className="mb-4 text-lg font-semibold">Structured State</h2>
              <pre className={`overflow-auto whitespace-pre-wrap break-words rounded-2xl p-4 text-xs leading-6 ${theme.soft}`}>
                {JSON.stringify(structuredState, null, 2) || "No structured state yet."}
              </pre>
            </div>

            <div className={`rounded-3xl border p-5 shadow-xl ${theme.card}`}>
              <h2 className="mb-4 text-lg font-semibold">Last API Response</h2>
              <pre className={`max-h-[360px] overflow-auto whitespace-pre-wrap break-words rounded-2xl p-4 text-xs leading-6 ${theme.soft}`}>
                {JSON.stringify(lastResponse, null, 2) || "No response yet."}
              </pre>
            </div>

            <div className={`rounded-3xl border p-5 shadow-xl ${theme.card}`}>
              <h2 className="mb-4 text-lg font-semibold">Text Sources</h2>
              <div className="space-y-3">
                {textMatches.length === 0 ? (
                  <p className={`text-sm ${theme.muted}`}>No text sources yet.</p>
                ) : (
                  textMatches.map((doc, idx) => (
                    <div key={idx} className={`rounded-2xl border p-3 text-xs leading-5 ${theme.soft}`}>
                      <div><strong>Source:</strong> {doc.source || "unknown"}</div>
                      <div><strong>Pages:</strong> {doc.page_start ?? "?"} - {doc.page_end ?? "?"}</div>
                      <div><strong>Section:</strong> {doc.section_title || "unknown"}</div>
                      <details className="mt-2">
                        <summary className="cursor-pointer text-xs font-semibold">Excerpt</summary>
                        <p className="mt-2 whitespace-pre-wrap">{shortText(doc.text)}</p>
                      </details>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className={`rounded-3xl border p-5 shadow-xl ${theme.card}`}>
              <h2 className="mb-4 text-lg font-semibold">Image Sources</h2>
              <div className="space-y-3">
                {imageMatches.length === 0 ? (
                  <p className={`text-sm ${theme.muted}`}>No image matches yet.</p>
                ) : (
                  imageMatches.map((item, idx) => (
                    <div key={idx} className={`rounded-2xl border p-3 text-xs leading-5 ${theme.soft}`}>
                      <div><strong>Source:</strong> {item.source || item.source_pdf || "unknown"}</div>
                      <div><strong>Page:</strong> {item.page ?? "?"}</div>
                      <div><strong>Score:</strong> {typeof item.score === "number" ? item.score.toFixed(4) : "n/a"}</div>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className={`rounded-3xl border p-5 shadow-xl ${theme.card}`}>
              <h2 className="mb-4 text-lg font-semibold">Classifier (System B)</h2>
              <pre className={`overflow-auto whitespace-pre-wrap break-words rounded-2xl p-4 text-xs leading-6 ${theme.soft}`}>
                {JSON.stringify(classifierResult, null, 2) || "No classifier output yet."}
              </pre>
            </div>

            <div className={`rounded-3xl border p-5 shadow-xl ${theme.card}`}>
              <h2 className="mb-4 text-lg font-semibold">Retrieval Debug</h2>
              <pre className={`overflow-auto whitespace-pre-wrap break-words rounded-2xl p-4 text-xs leading-6 ${theme.soft}`}>
                {JSON.stringify(retrievalDebug, null, 2) || "No retrieval debug yet."}
              </pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}