import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  sendChatMessage,
  fetchConversations,
  createConversation,
  fetchConversation,
  deleteConversation,
} from './api';

const CAT_ICONS = {
  Shopping: '🛒',
  Travel: '✈️',
  Education: '📚',
  Payment: '💳',
  Communication: '💬',
  Work_Career: '💼',
  'Work & Career': '💼',
  Entertainment: '🎮',
  Connectivity: '📶',
  General: '📁',
};

const CAT_COLORS = {
  Shopping: '#f59e0b',
  Travel: '#06b6d4',
  Education: '#8b5cf6',
  Payment: '#10b981',
  Communication: '#3b82f6',
  Work_Career: '#6366f1',
  'Work & Career': '#6366f1',
  Entertainment: '#ec4899',
  Connectivity: '#f97316',
  General: '#9ca3af',
};

function catIcon(name) { return CAT_ICONS[name] || '🗂️'; }
function catColor(name) { return CAT_COLORS[name] || '#64748b'; }

function ChatSpinner() {
  return (
    <div className="chat-loading-dots">
      <span className="dot dot-1" />
      <span className="dot dot-2" />
      <span className="dot dot-3" />
    </div>
  );
}

// ── Simple Markdown Text Formatter ───────────────────────────────────────────
function FormattedMessage({ text }) {
  if (!text) return null;

  const lines = text.split('\n');
  return (
    <div className="chat-message-text">
      {lines.map((line, idx) => {
        if (!line.trim()) {
          return <div key={idx} className="chat-text-gap" />;
        }

        const isBullet = line.trim().startsWith('- ') || line.trim().startsWith('• ');
        const isQuote = line.trim().startsWith('> ');

        let content = line.trim();
        if (isBullet) content = content.replace(/^[-•]\s*/, '');
        if (isQuote) content = content.replace(/^>\s*/, '');

        // Format bold (**...**) and code (`...`)
        const parts = [];
        const regex = /(\*\*.*?\*\*|`.*?`)/g;
        let lastIndex = 0;
        let match;

        while ((match = regex.exec(content)) !== null) {
          if (match.index > lastIndex) {
            parts.push(content.substring(lastIndex, match.index));
          }
          const raw = match[0];
          if (raw.startsWith('**') && raw.endsWith('**')) {
            parts.push(<strong key={match.index}>{raw.slice(2, -2)}</strong>);
          } else if (raw.startsWith('`') && raw.endsWith('`')) {
            parts.push(<code key={match.index} className="chat-inline-code">{raw.slice(1, -1)}</code>);
          }
          lastIndex = regex.lastIndex;
        }
        if (lastIndex < content.length) {
          parts.push(content.substring(lastIndex));
        }

        if (isBullet) {
          return (
            <div key={idx} className="chat-bullet-line">
              <span className="chat-bullet-marker">•</span>
              <span>{parts}</span>
            </div>
          );
        }

        if (isQuote) {
          return (
            <blockquote key={idx} className="chat-quote-line">
              {parts}
            </blockquote>
          );
        }

        return <p key={idx} className="chat-para-line">{parts}</p>;
      })}
    </div>
  );
}

// ── Screenshot Evidence Card ──────────────────────────────────────────────────
function ScreenshotReferenceCard({ item, onSelect, onTagClick }) {
  const color = catColor(item.category);
  const tags = item.tags || [];

  return (
    <div
      className="chat-ref-card"
      onClick={() => onSelect?.(item)}
      title="Click to view full screenshot details"
      style={{ borderColor: color + '44' }}
    >
      <div className="chat-ref-thumb-wrap">
        {item.image_url ? (
          <img
            src={item.image_url}
            alt={item.filename || item.screenshot_id}
            className="chat-ref-thumb"
            onError={(e) => {
              e.target.style.display = 'none';
              e.target.nextSibling.style.display = 'flex';
            }}
          />
        ) : null}
        <div
          className="chat-ref-placeholder"
          style={{ display: item.image_url ? 'none' : 'flex', color }}
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <circle cx="8.5" cy="8.5" r="1.5" />
            <polyline points="21 15 16 10 5 21" />
          </svg>
        </div>
        <div className="chat-ref-score" style={{ background: color + '22', color, borderColor: color + '55' }}>
          {(item.similarity_score * 100).toFixed(0)}% MATCH
        </div>
      </div>

      <div className="chat-ref-info">
        <div className="chat-ref-header">
          <span className="chat-ref-id">[{item.screenshot_id}]</span>
          <span className="chat-ref-badge" style={{ color, borderColor: color + '55', background: color + '15' }}>
            {catIcon(item.category)} {item.category}
          </span>
        </div>
        <div className="chat-ref-filename">{item.filename}</div>

        {item.ocr_excerpt && (
          <div className="chat-ref-excerpt">
            "{item.ocr_excerpt}"
          </div>
        )}

        {tags.length > 0 && (
          <div className="chat-ref-tags">
            {tags.slice(0, 3).map((t, idx) => (
              <span
                key={idx}
                className="chat-ref-tag"
                onClick={(e) => {
                  e.stopPropagation();
                  onTagClick?.(t);
                }}
                title={`Filter by tag: ${t}`}
              >
                #{t}
              </span>
            ))}
            {tags.length > 3 && (
              <span className="chat-ref-tag more">+{tags.length - 3}</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

const STORAGE_KEY = 'iris_active_conversation_id';
const WELCOME_MESSAGE = {
  id: 'welcome',
  sender: 'assistant',
  text: "Hello! I am **I.R.I.S.**, your Intelligent Screenshot Assistant. Ask me anything about your screenshot library, receipts, travel tickets, visiting cards, or notes.",
  references: [],
};

// ── Main ChatTab Component ───────────────────────────────────────────────────
export default function ChatTab({ onSelect, onTagClick }) {
  const [conversationId, setConversationId] = useState(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) || null;
    } catch (_) {
      return null;
    }
  });
  const [conversations, setConversations] = useState([]);
  const [messages, setMessages] = useState([WELCOME_MESSAGE]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [restoring, setRestoring] = useState(false);
  const [error, setError] = useState(null);
  const [lastQuery, setLastQuery] = useState('');

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const loadConversationList = useCallback(async () => {
    try {
      const data = await fetchConversations();
      setConversations(Array.isArray(data) ? data : []);
    } catch (e) {
      // Non-fatal if offline
    }
  }, []);

  const loadConversation = useCallback(async (id) => {
    if (!id) {
      setConversationId(null);
      setMessages([WELCOME_MESSAGE]);
      try { localStorage.removeItem(STORAGE_KEY); } catch (_) {}
      return;
    }

    setRestoring(true);
    setError(null);
    try {
      const data = await fetchConversation(id);
      if (data && Array.isArray(data.messages) && data.messages.length > 0) {
        const restored = data.messages.map((m) => ({
          id: m.message_id || `msg-${Math.random()}`,
          sender: m.role === 'user' ? 'user' : 'assistant',
          text: m.content,
          references: m.references || [],
        }));
        setMessages(restored);
      } else {
        setMessages([WELCOME_MESSAGE]);
      }
      setConversationId(id);
      try { localStorage.setItem(STORAGE_KEY, id); } catch (_) {}
    } catch (err) {
      setConversationId(null);
      setMessages([WELCOME_MESSAGE]);
      try { localStorage.removeItem(STORAGE_KEY); } catch (_) {}
    } finally {
      setRestoring(false);
    }
  }, []);

  // Initial load on mount (restores active session across tab navigation & refreshes)
  useEffect(() => {
    loadConversationList();
    const savedId = (() => {
      try { return localStorage.getItem(STORAGE_KEY); } catch (_) { return null; }
    })();
    if (savedId) {
      loadConversation(savedId);
    }
  }, [loadConversationList, loadConversation]);

  const handleNewChat = async () => {
    try {
      const res = await createConversation("New Chat");
      const newId = res.conversation_id;
      setConversationId(newId);
      try { localStorage.setItem(STORAGE_KEY, newId); } catch (_) {}
      setMessages([WELCOME_MESSAGE]);
      setError(null);
      loadConversationList();
      setTimeout(() => inputRef.current?.focus(), 50);
    } catch (_) {
      setConversationId(null);
      try { localStorage.removeItem(STORAGE_KEY); } catch (_) {}
      setMessages([WELCOME_MESSAGE]);
    }
  };

  const handleDeleteChat = async (idToDelete) => {
    const targetId = idToDelete || conversationId;
    if (!targetId) return;
    if (!window.confirm("Delete this conversation and its message history?")) return;
    try {
      await deleteConversation(targetId);
      if (targetId === conversationId) {
        setConversationId(null);
        try { localStorage.removeItem(STORAGE_KEY); } catch (_) {}
        setMessages([WELCOME_MESSAGE]);
      }
      loadConversationList();
    } catch (err) {
      alert(`Failed to delete conversation: ${err.message}`);
    }
  };

  const handleSend = async (queryText) => {
    const query = (queryText || input).trim();
    if (!query || loading) return;

    setLastQuery(query);
    setError(null);
    setInput('');

    const userMessage = {
      id: `user-${Date.now()}`,
      sender: 'user',
      text: query,
    };

    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);

    try {
      const recentHistory = messages
        .filter((m) => m.id !== 'welcome')
        .slice(-6)
        .map((m) => ({
          role: m.sender === 'user' ? 'user' : 'assistant',
          content: m.text,
          referenced_ids: m.references ? m.references.map((r) => r.screenshot_id) : [],
        }));

      const data = await sendChatMessage(query, {
        top_k: 3,
        conversation_id: conversationId,
        history: recentHistory,
      });

      if (data.conversation_id && data.conversation_id !== conversationId) {
        setConversationId(data.conversation_id);
        try { localStorage.setItem(STORAGE_KEY, data.conversation_id); } catch (_) {}
        loadConversationList();
      }

      const assistantMessage = {
        id: `asst-${Date.now()}`,
        sender: 'assistant',
        text: data.reply,
        references: data.referenced_screenshots || [],
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const errMsg = err.message || 'Failed to communicate with I.R.I.S. backend.';
      setError(errMsg);

      const errorMessage = {
        id: `err-${Date.now()}`,
        sender: 'assistant',
        text: `⚠️ **Error:** ${errMsg}`,
        isError: true,
      };

      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleRetry = () => {
    if (lastQuery) {
      handleSend(lastQuery);
    }
  };

  const SUGGESTED_QUERIES = [
    'Find my visiting card',
    'Show my flight or train tickets',
    'Find credit card or bank statements',
    'Find recharge or bill receipts',
  ];

  return (
    <div className="iris-chat-container">
      {/* ── Conversation Session Toolbar ── */}
      <div className="iris-chat-session-bar">
        <div className="chat-session-left">
          <button
            type="button"
            className="chat-session-btn chat-new-chat-btn"
            id="iris-new-chat-btn"
            onClick={handleNewChat}
            title="Start a new conversation"
          >
            <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            <span>NEW CHAT</span>
          </button>

          {conversations.length > 0 && (
            <div className="chat-session-picker-wrap">
              <span className="chat-session-label">CHAT:</span>
              <select
                className="chat-session-select"
                id="iris-chat-session-select"
                value={conversationId || ''}
                onChange={(e) => {
                  const val = e.target.value;
                  if (val) loadConversation(val);
                }}
              >
                {!conversationId && <option value="">Active (Unsaved session)</option>}
                {conversations.map((c) => (
                  <option key={c.conversation_id} value={c.conversation_id}>
                    {c.title} ({c.message_count || 0})
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        <div className="chat-session-right">
          {conversationId && (
            <button
              type="button"
              className="chat-session-delete-btn"
              onClick={() => handleDeleteChat()}
              title="Delete this conversation"
            >
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
              </svg>
              <span>DELETE CHAT</span>
            </button>
          )}
          {restoring && <span className="chat-session-restoring">Restoring history…</span>}
        </div>
      </div>
      {/* ── Chat Messages Scroll Area ── */}
      <div className="iris-chat-messages" id="chat-messages-container">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`chat-bubble-row ${msg.sender === 'user' ? 'user' : 'assistant'}`}
          >
            {msg.sender === 'assistant' && (
              <div className="chat-avatar assistant-avatar" title="I.R.I.S. Assistant">
                <svg viewBox="0 0 100 100" width="22" height="22">
                  <circle cx="50" cy="50" r="42" fill="none" stroke="#00d4ff" strokeWidth="6" />
                  <circle cx="50" cy="50" r="26" fill="none" stroke="#0066ff" strokeWidth="8" strokeDasharray="20 10" />
                  <circle cx="50" cy="50" r="12" fill="#00f5d4" />
                </svg>
              </div>
            )}

            <div className={`chat-bubble ${msg.sender} ${msg.isError ? 'error' : ''}`}>
              <FormattedMessage text={msg.text} />

              {/* Render Evidence References if available */}
              {msg.references && msg.references.length > 0 && (
                <div className="chat-references-container">
                  <div className="chat-references-header">
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
                      <rect x="3" y="3" width="18" height="18" rx="2" />
                      <circle cx="8.5" cy="8.5" r="1.5" />
                      <polyline points="21 15 16 10 5 21" />
                    </svg>
                    <span>SUPPORTING EVIDENCE ({msg.references.length})</span>
                  </div>

                  <div className="chat-references-grid">
                    {msg.references.map((ref) => (
                      <ScreenshotReferenceCard
                        key={ref.screenshot_id}
                        item={ref}
                        onSelect={onSelect}
                        onTagClick={onTagClick}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>

            {msg.sender === 'user' && (
              <div className="chat-avatar user-avatar" title="You">
                <span>U</span>
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="chat-bubble-row assistant">
            <div className="chat-avatar assistant-avatar">
              <svg viewBox="0 0 100 100" width="22" height="22">
                <circle cx="50" cy="50" r="42" fill="none" stroke="#00d4ff" strokeWidth="6" />
                <circle cx="50" cy="50" r="12" fill="#00f5d4" />
              </svg>
            </div>
            <div className="chat-bubble assistant loading">
              <div className="chat-loading-wrapper">
                <ChatSpinner />
                <span className="chat-loading-label">Searching screenshot library & evaluating OCR evidence…</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* ── Suggested Prompts (when only welcome message is shown) ── */}
      {messages.length === 1 && !loading && (
        <div className="chat-suggested-prompts">
          <div className="chat-suggested-title">Suggested Inquiries:</div>
          <div className="chat-suggested-chips">
            {SUGGESTED_QUERIES.map((q, idx) => (
              <button
                key={idx}
                type="button"
                className="chat-suggested-btn"
                onClick={() => handleSend(q)}
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ── Error Banner & Retry ── */}
      {error && (
        <div className="chat-error-bar">
          <span>{error}</span>
          <button type="button" className="chat-retry-btn" onClick={handleRetry}>
            RETRY
          </button>
        </div>
      )}

      {/* ── Chat Input Footer ── */}
      <div className="iris-chat-input-bar">
        <div className="iris-chat-input-wrapper">
          <input
            ref={inputRef}
            id="iris-chat-input"
            type="text"
            className="iris-chat-input-field"
            placeholder="Ask I.R.I.S. about your screenshots… (e.g., 'Find Kanpur visiting card')"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
            autoFocus
          />
          <button
            type="button"
            id="iris-chat-send-btn"
            className="iris-chat-send-btn"
            onClick={() => handleSend()}
            disabled={loading || !input.trim()}
            title="Send message"
          >
            {loading ? (
              <ChatSpinner />
            ) : (
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
