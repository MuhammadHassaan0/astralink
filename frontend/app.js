(() => {
  const STORAGE_KEYS = {
    session: "astralink_session_id",
    profile: "astralink_profile",
    auth: "astralink_auth_token",
  };
  const USER_ID_KEY = "astralink_user_id";

  const REVEAL_STYLE = `
    [data-animate] {
      opacity: 0;
      transform: translateY(18px) scale(0.99);
      transition: opacity 0.7s ease, transform 0.7s ease;
    }
    [data-animate].is-visible {
      opacity: 1;
      transform: translateY(0) scale(1);
    }
  `;
  let animationStyleInjected = false;

  function ensureAnimationStyles() {
    if (animationStyleInjected) return;
    const style = document.createElement("style");
    style.textContent = REVEAL_STYLE;
    document.head.appendChild(style);
    animationStyleInjected = true;
  }

  function initRevealAnimations() {
    if (typeof IntersectionObserver === "undefined") {
      document.querySelectorAll("[data-animate]").forEach((el) => el.classList.add("is-visible"));
      return;
    }
    ensureAnimationStyles();
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.16, rootMargin: "0px 0px -10% 0px" });

    document.querySelectorAll("[data-animate]").forEach((el, idx) => {
      if (el.dataset.animateDelay === undefined) {
        const delay = Math.min(idx * 80, 360);
        el.style.transitionDelay = `${delay}ms`;
      }
      observer.observe(el);
    });
  }

  function getSessionId() {
    return localStorage.getItem(STORAGE_KEYS.session) || "";
  }

  function setSessionId(id) {
    if (id) {
      localStorage.setItem(STORAGE_KEYS.session, id);
    } else {
      localStorage.removeItem(STORAGE_KEYS.session);
    }
  }

  function getAuthToken() {
    return localStorage.getItem(STORAGE_KEYS.auth) || "";
  }

  function setAuthToken(token) {
    if (token) {
      localStorage.setItem(STORAGE_KEYS.auth, token);
    } else {
      localStorage.removeItem(STORAGE_KEYS.auth);
    }
  }

  function isAuthenticated() {
    return Boolean(getAuthToken());
  }

  function getProfile() {
    try {
      const raw = localStorage.getItem(STORAGE_KEYS.profile);
      return raw ? JSON.parse(raw) : {};
    } catch (err) {
      console.warn("profile parse failed", err);
      return {};
    }
  }

  function generateUserId() {
    if (typeof crypto !== "undefined" && crypto.randomUUID) {
      return crypto.randomUUID();
    }
    return `user_${Date.now()}_${Math.random().toString(16).slice(2)}`;
  }

  function getUserId() {
    let existing = localStorage.getItem(USER_ID_KEY);
    if (!existing) {
      existing = generateUserId();
      localStorage.setItem(USER_ID_KEY, existing);
    }
    return existing;
  }

  function getPersonaId() {
    return getSessionId() || getUserId();
  }

  function saveProfileLocal(profile) {
    localStorage.setItem(STORAGE_KEYS.profile, JSON.stringify(profile || {}));
  }

  function normalizeMode(value) {
    const v = (value || "").toLowerCase();
    if (v.includes("alive") || v.includes("unavailable")) return "alive";
    return "memory";
  }

  function buildHeaders(existing) {
    const headers = existing ? { ...existing } : {};
    const token = getAuthToken();
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    const sessionId = getSessionId();
    if (sessionId) {
      headers["X-Astralink-Session"] = sessionId;
    }
    headers["Accept"] = headers["Accept"] || "application/json";
    return headers;
  }

  async function authFetch(path, options = {}) {
    const opts = { ...options };
    opts.headers = buildHeaders(opts.headers);
    if (opts.json !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(opts.json);
      delete opts.json;
    }
    const res = await fetch(path, opts);
    const raw = await res.text();
    let data;
    try {
      data = raw ? JSON.parse(raw) : {};
    } catch (err) {
      throw new Error("Unexpected response");
    }
    if (data && data.session_id) {
      setSessionId(data.session_id);
    }
    if (!res.ok || data.error) {
      throw new Error(data.error || "Request failed");
    }
    return data;
  }

  function handleAuthResponse(data) {
    if (data.session_id) {
      setSessionId(data.session_id);
    }
    if (data.auth_token) {
      setAuthToken(data.auth_token);
    }
    if (data.profile) {
      saveProfileLocal(data.profile);
    }
  }

  async function saveProfile(profile) {
    const payload = {
      session: getSessionId(),
      session_id: getSessionId(),
      profile,
    };
    const data = await authFetch("/api/save_profile", {
      method: "POST",
      json: payload,
    });
    if (data.session_id) setSessionId(data.session_id);
    saveProfileLocal(profile);
    return data;
  }

  async function postJSON(path, payload) {
    return authFetch(path, {
      method: "POST",
      json: payload,
    });
  }

  async function sendChatRequest(payload) {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const raw = await res.text();
    let data = {};
    try {
      data = raw ? JSON.parse(raw) : {};
    } catch (err) {
      console.error("chat parse failed", err, raw);
      throw new Error("Chat response parse error");
    }
    if (!res.ok || data.error) {
      throw new Error(data.error || "Chat request failed");
    }
    return data;
  }

  async function chat(input) {
    const candidate =
      input && typeof input === "object" && !Array.isArray(input) ? input.messages : undefined;
    const history = Array.isArray(candidate)
      ? candidate
      : [
          {
            role: "user",
            content:
              typeof input === "string"
                ? input
                : String(
                    (input && typeof input === "object" && "message" in input ? input.message : "") ||
                      (input && typeof input === "object" && "content" in input ? input.content : "")
                  ),
          },
        ];
    return sendChatRequest({
      userId: getUserId(),
      personaId: getPersonaId(),
      messages: history,
    });
  }

  async function listConversations() {
    const sessionId = getSessionId();
    const query = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
    const data = await authFetch(`/api/conversations${query}`);
    return data.conversations || [];
  }

  async function createConversation(title) {
    const payload = title ? { title } : {};
    payload.session_id = getSessionId();
    const data = await authFetch("/api/conversations", {
      method: "POST",
      json: payload,
    });
    return data.conversation;
  }

  async function getConversation(convoId) {
    const sessionId = getSessionId();
    const query = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
    const data = await authFetch(`/api/conversations/${convoId}${query}`);
    return data.conversation;
  }

  async function sendConversationMessage({ userId, personaId, messages }) {
    if (!Array.isArray(messages) || !messages.length) {
      throw new Error("messages array required");
    }
    return sendChatRequest({
      userId: userId || getUserId(),
      personaId: personaId || getPersonaId(),
      messages,
    });
  }

  async function renameConversation(convoId, title) {
    const data = await authFetch(`/api/conversations/${convoId}/title`, {
      method: "POST",
      json: { title, session_id: getSessionId() },
    });
    return data.conversation;
  }

  function requireSession() {
    const sid = getSessionId();
    if (!sid) throw new Error("Please save their profile first.");
    return sid;
  }

  async function endSession() {
    const sid = requireSession();
    return postJSON("/api/end", { session_id: sid });
  }

  async function signupAccount(name, email, password) {
    const data = await authFetch("/api/signup", {
      method: "POST",
      json: { name, email, password },
    });
    handleAuthResponse(data);
    return data;
  }

  async function loginAccount(email, password) {
    const data = await authFetch("/api/login", {
      method: "POST",
      json: { email, password },
    });
    handleAuthResponse(data);
    return data;
  }

  async function logoutAccount() {
    try {
      await authFetch("/api/logout", { method: "POST" });
    } catch (err) {
      // ignore
    }
    setAuthToken(null);
    setSessionId("");
  }

  window.Astralink = {
    STORAGE_KEYS,
    getSessionId,
    setSessionId,
    getAuthToken,
    setAuthToken,
    getUserId,
    getPersonaId,
    isAuthenticated,
    getProfile,
    saveProfileLocal,
    saveProfile,
    postJSON,
    chat,
    endSession,
    signupAccount,
    loginAccount,
    logoutAccount,
    handleAuthResponse,
    authFetch,
    requireSession,
    normalizeMode,
    initRevealAnimations,
    listConversations,
    createConversation,
    getConversation,
    sendConversationMessage,
    renameConversation,
  };

  document.addEventListener("DOMContentLoaded", () => {
    initRevealAnimations();
  });
})();
