import { useEffect, useRef, useState } from 'react'

const ASSET_IMAGE = '/gsam-bg.png'
const PAGE_ACTION_PREFIX = '__PAGE_ACTION__:'
const AUTH_STORAGE_KEY = 'gsam-auth'
const SESSION_STORAGE_KEY = 'gsam-active-session'

function AuthPanel({ backendCandidates, onAuthenticated }) {
  const [mode, setMode] = useState('login')
  const [backendUrl, setBackendUrl] = useState(backendCandidates[0] || 'http://127.0.0.1:8000')
  const [form, setForm] = useState({ username: '', email: '', password: '' })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const submit = async (event) => {
    event.preventDefault()
    setError('')
    setLoading(true)
    try {
      const endpoint = mode === 'register' ? '/auth/register' : '/auth/login'
      const payload = mode === 'register'
        ? { username: form.username, email: form.email, password: form.password }
        : { email: form.email, password: form.password }

      const response = await fetch(`${backendUrl}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(data.detail || 'Connexion impossible')

      onAuthenticated({ backendUrl, user: data.user, token: data.token })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen w-full overflow-hidden text-white relative flex items-center justify-center px-4">
      <div
        className="fixed inset-0 z-0"
        style={{
          backgroundImage: `url(${ASSET_IMAGE})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
        }}
      />
      <div className="fixed inset-0 z-0 bg-black/70" />

      <form onSubmit={submit} className="relative z-10 w-full max-w-sm bg-black/35 backdrop-blur-xl border border-white/15 rounded-xl p-6 shadow-2xl">
        <div className="flex items-center gap-3 mb-6">
          <img
            src={ASSET_IMAGE}
            alt="GSAM"
            className="h-12 w-12 rounded-full object-cover border-2 border-[#f2ca50]/50"
          />
          <div>
            <h1 className="text-[#f2ca50] font-black tracking-widest text-sm">GSAM</h1>
            <p className="text-[10px] uppercase tracking-widest text-white/45">
              {mode === 'login' ? 'Connexion securisee' : 'Creation de compte'}
            </p>
          </div>
        </div>

        <div className="flex gap-2 mb-5 bg-white/5 border border-white/10 rounded-lg p-1">
          <button
            type="button"
            onClick={() => { setMode('login'); setError('') }}
            className={`flex-1 py-2 text-[10px] uppercase tracking-widest rounded ${mode === 'login' ? 'bg-[#f2ca50] text-[#3d2f00] font-bold' : 'text-white/55 hover:text-white'}`}
          >
            Connexion
          </button>
          <button
            type="button"
            onClick={() => { setMode('register'); setError('') }}
            className={`flex-1 py-2 text-[10px] uppercase tracking-widest rounded ${mode === 'register' ? 'bg-[#f2ca50] text-[#3d2f00] font-bold' : 'text-white/55 hover:text-white'}`}
          >
            Inscription
          </button>
        </div>

        <div className="space-y-3">
          {mode === 'register' && (
            <input
              value={form.username}
              onChange={e => setForm(prev => ({ ...prev, username: e.target.value }))}
              required
              className="w-full bg-white/10 border border-white/15 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:border-[#f2ca50]/60 placeholder:text-white/35"
              placeholder="Nom utilisateur"
            />
          )}
          <input
            type="email"
            value={form.email}
            onChange={e => setForm(prev => ({ ...prev, email: e.target.value }))}
            required
            className="w-full bg-white/10 border border-white/15 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:border-[#f2ca50]/60 placeholder:text-white/35"
            placeholder="Email"
          />
          <input
            type="password"
            value={form.password}
            onChange={e => setForm(prev => ({ ...prev, password: e.target.value }))}
            required
            minLength={8}
            className="w-full bg-white/10 border border-white/15 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:border-[#f2ca50]/60 placeholder:text-white/35"
            placeholder="Mot de passe"
          />
          <select
            value={backendUrl}
            onChange={e => setBackendUrl(e.target.value)}
            className="w-full bg-black/60 border border-white/15 rounded-lg px-3 py-2.5 text-xs text-white/70 focus:outline-none focus:border-[#60a5fa]/60"
          >
            {backendCandidates.map(url => <option key={url} value={url}>{url}</option>)}
          </select>
        </div>

        {error && (
          <p className="mt-4 text-xs text-[#ffb4ab] border border-[#ffb4ab]/20 bg-[#ffb4ab]/10 rounded-lg px-3 py-2">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={loading}
          className="mt-5 w-full bg-gradient-to-r from-[#f2ca50] to-[#d4af37] text-[#3d2f00] font-black text-[10px] uppercase tracking-widest py-3 rounded-lg disabled:opacity-50"
        >
          {loading ? 'Veuillez patienter...' : mode === 'login' ? 'Se connecter' : 'Creer le compte'}
        </button>
      </form>
    </div>
  )
}

function formatSessionTime(value) {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleString('fr-FR', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })
}

function mapStoredMessage(message) {
  const time = formatSessionTime(message.timestamp)
  if (message.role === 'user') {
    const text = message.content?.startsWith(PAGE_ACTION_PREFIX)
      ? message.content.slice(PAGE_ACTION_PREFIX.length).trim()
      : message.content
    return { id: message.id, type: 'user', text, time }
  }
  if (message.messageType === 'result') {
    let data = {}
    try {
      data = JSON.parse(message.content || '{}')
    } catch (_) {
      data = { resultat: message.content }
    }
    return { id: message.id, type: 'result', data }
  }
  if (message.messageType === 'error') {
    return { id: message.id, type: 'action', name: 'ERROR', args: message.content, status: 'error' }
  }
  if (message.messageType === 'action') {
    return { id: message.id, type: 'action', name: message.content, status: 'done' }
  }
  return { id: message.id, type: 'question', text: message.content }
}

function SessionSidebar({ sessions, activeSessionId, onSelect, onNew, onDelete, loading, user }) {
  return (
    <aside className="hidden lg:flex flex-col w-72 bg-black/35 backdrop-blur-md border-r border-white/10 relative z-10">
      <div className="p-3 border-b border-white/10">
        <p className="text-[9px] uppercase tracking-widest text-white/35 mb-1">Utilisateur</p>
        <div className="flex items-center gap-2 min-w-0">
          <span className="material-symbols-outlined text-[#f2ca50] text-base">account_circle</span>
          <p className="text-xs font-bold text-white/80 truncate">{user?.username || user?.email}</p>
        </div>
      </div>
      <div className="p-3 border-b border-white/10 flex items-center gap-2">
        <span className="material-symbols-outlined text-[#f2ca50] text-sm">forum</span>
        <p className="text-[10px] uppercase tracking-widest text-[#f2ca50] font-bold">Sujets</p>
        <button
          onClick={onNew}
          className="ml-auto text-[9px] uppercase tracking-widest text-[#4ade80] border border-[#4ade80]/25 px-2 py-1 rounded hover:bg-[#4ade80]/10"
        >
          Nouveau
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-2">
        {loading ? (
          <p className="text-[10px] text-white/35 px-2 py-3">Chargement...</p>
        ) : sessions.length === 0 ? (
          <p className="text-[10px] text-white/35 px-2 py-3">Aucun sujet pour le moment.</p>
        ) : sessions.map(session => (
          <div key={session.id} className="relative group">
            <button
              onClick={() => onSelect(session.id)}
              className={`w-full text-left rounded-lg border p-3 transition-colors ${activeSessionId === session.id ? 'border-[#f2ca50]/50 bg-[#f2ca50]/10' : 'border-white/10 bg-white/[0.03] hover:bg-white/[0.07]'}`}
            >
              <div className="flex items-center gap-2 mb-1">
                <span className={`h-1.5 w-1.5 rounded-full ${session.status === 'running' ? 'bg-[#60a5fa]' : session.status === 'completed' ? 'bg-[#4ade80]' : 'bg-white/35'}`} />
                <p className="text-xs font-bold text-white/85 truncate pr-6">{session.task || 'Sujet sans titre'}</p>
              </div>
              <p className="text-[10px] text-white/45 line-clamp-2 min-h-[2rem]">
                {session.lastMessage || 'Conversation sauvegardee'}
              </p>
              <p className="text-[9px] uppercase tracking-widest text-white/30 mt-2">
                {formatSessionTime(session.lastMessageAt || session.startedAt)}
              </p>
            </button>
            <button
              onClick={(e) => { e.stopPropagation(); onDelete(session.id); }}
              className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-red-500/20 border border-red-500/30"
              title="Supprimer cette session"
            >
              <span className="material-symbols-outlined text-red-400 text-sm">delete</span>
            </button>
          </div>
        ))}
      </div>
    </aside>
  )
}

function FeedbackInput({ question, onSubmit }) {
  const [text, setText] = useState('')

  // Detect select options: "Label (options: A, B, C) :"
  const selectMatch = question?.match(/\(options:\s*(.+?)\)\s*:?\s*$/i)
  const selectOptions = selectMatch
    ? selectMatch[1].split(',').map(o => o.trim()).filter(Boolean)
    : null

  // Select field: show clickable option buttons
  if (selectOptions && selectOptions.length > 0) {
    return (
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-2 max-h-72 overflow-y-auto pr-1">
          {selectOptions.map(opt => (
            <button
              key={opt}
              onClick={() => onSubmit(opt)}
              className="py-2.5 px-3 text-left text-sm text-white bg-white/5 border border-white/15 rounded-lg hover:bg-[#f2ca50]/20 hover:border-[#f2ca50]/50 transition-colors"
            >
              {opt}
            </button>
          ))}
        </div>
      </div>
    )
  }

  // Regular text input
  return (
    <div className="space-y-3">
      <textarea
        autoFocus
        rows={3}
        value={text}
        onChange={e => setText(e.target.value)}
        onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && text.trim()) { e.preventDefault(); onSubmit(text.trim()) } }}
        className="w-full bg-white/10 border border-white/20 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-[#f2ca50]/60 resize-none placeholder:text-white/40"
        placeholder="Votre réponse... (Enter pour envoyer)"
      />
      <div className="flex gap-2">
        <button onClick={() => text.trim() && onSubmit(text.trim())}
          className="w-full py-2 text-[10px] uppercase tracking-widest font-bold bg-gradient-to-r from-[#f2ca50] to-[#d4af37] text-[#3d2f00] rounded">SEND</button>
      </div>
    </div>
  )
}

function ActionableFeedbackInput({ question, onSubmit, onSkip }) {
  const [text, setText] = useState('')
  const [mode, setMode] = useState('answer')

  const submitValue = (value) => {
    const trimmed = value.trim()
    if (!trimmed) return
    onSubmit(mode === 'redirect' ? `${PAGE_ACTION_PREFIX}${trimmed}` : trimmed)
  }

  const selectMatch = question?.match(/\(options:\s*(.+?)\)\s*:?\s*$/i)
  const selectOptions = selectMatch
    ? selectMatch[1].split(',').map(o => o.trim()).filter(Boolean)
    : null

  if (selectOptions && selectOptions.length > 0 && mode !== 'redirect') {
    return (
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-2 max-h-72 overflow-y-auto pr-1">
          {selectOptions.map(opt => (
            <button
              key={opt}
              onClick={() => onSubmit(opt)}
              className="py-2.5 px-3 text-left text-sm text-white bg-white/5 border border-white/15 rounded-lg hover:bg-[#f2ca50]/20 hover:border-[#f2ca50]/50 transition-colors"
            >
              {opt}
            </button>
          ))}
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => { setMode('redirect'); setText('') }}
            className="flex-1 py-2 text-[10px] uppercase tracking-widest text-[#60a5fa] border border-[#60a5fa]/25 rounded hover:bg-[#60a5fa]/10"
          >
            AUTRE ACTION
          </button>
          <button
            onClick={onSkip}
            className="flex-1 py-2 text-[10px] uppercase tracking-widest text-white/60 border border-white/20 rounded hover:bg-white/10"
          >
            SKIP
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {mode === 'redirect' && (
        <p className="text-[11px] text-[#60a5fa] uppercase tracking-widest">
          Donne une autre action à faire sur la page courante
        </p>
      )}
      <textarea
        autoFocus
        rows={3}
        value={text}
        onChange={e => setText(e.target.value)}
        onKeyDown={e => {
          if (e.key === 'Enter' && !e.shiftKey && text.trim()) {
            e.preventDefault()
            submitValue(text)
          }
        }}
        className={`w-full rounded px-3 py-2 text-sm text-white focus:outline-none resize-none placeholder:text-white/40 ${mode === 'redirect' ? 'bg-white/10 border border-[#60a5fa]/30 focus:border-[#60a5fa]/70' : 'bg-white/10 border border-white/20 focus:border-[#f2ca50]/60'}`}
        placeholder={mode === 'redirect' ? 'Ex: ouvre le menu, clique sur retour, cherche un autre bouton...' : 'Votre réponse... (Enter pour envoyer)'}
      />
      <div className="flex gap-2">
        <button
          onClick={() => { setMode(mode === 'redirect' ? 'answer' : 'redirect'); setText('') }}
          className="flex-1 py-2 text-[10px] uppercase tracking-widest text-[#60a5fa] border border-[#60a5fa]/25 rounded hover:bg-[#60a5fa]/10"
        >
          {mode === 'redirect' ? 'RÉPONSE NORMALE' : 'AUTRE ACTION'}
        </button>
        <button
          onClick={onSkip}
          className="px-3 py-2 text-[10px] uppercase tracking-widest text-white/60 border border-white/20 rounded hover:bg-white/10"
        >
          SKIP
        </button>
        <button
          onClick={() => submitValue(text)}
          className={`flex-1 py-2 text-[10px] uppercase tracking-widest font-bold rounded ${mode === 'redirect' ? 'bg-gradient-to-r from-[#60a5fa] to-[#3b82f6] text-white' : 'bg-gradient-to-r from-[#f2ca50] to-[#d4af37] text-[#3d2f00]'}`}
        >
          {mode === 'redirect' ? 'ENVOYER ACTION' : 'SEND'}
        </button>
      </div>
    </div>
  )
}

function AgentMessage({ msg, onRerun }) {
  if (msg.type === 'action') {
    return (
      <div className="flex items-start gap-3 py-2">
        <span className="material-symbols-outlined text-[#f2ca50] text-base mt-0.5">
          {msg.status === 'done' ? 'check_circle' : msg.status === 'running' ? 'pending' : 'radio_button_unchecked'}
        </span>
        <div>
          <p className="text-[11px] uppercase tracking-widest font-bold text-[#f2ca50] drop-shadow">{msg.name}</p>
          {msg.args && <p className="text-[11px] text-white/50 mt-0.5">{msg.args}</p>}
        </div>
      </div>
    )
  }
  if (msg.type === 'result') {
    return (
      <div className="bg-black/40 backdrop-blur-sm border border-[#f2ca50]/20 rounded-xl p-5 my-3">
        <p className="text-[10px] text-[#f2ca50] font-bold uppercase tracking-widest mb-3">✓ Résultat</p>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(msg.data).map(([k, v]) => (
            <div key={k} className="bg-white/5 rounded p-3">
              <p className="text-[9px] text-white/40 uppercase tracking-widest mb-1">{k.replace(/_/g, ' ')}</p>
              <p className="text-sm font-bold text-[#f2ca50] break-all">{String(v)}</p>
            </div>
          ))}
        </div>
      </div>
    )
  }
  if (msg.type === 'question') {
    return (
      <div className="bg-black/40 backdrop-blur-sm border border-[#f2ca50]/30 rounded-xl p-4 my-3">
        <div className="flex items-center gap-2 mb-2">
          <span className="material-symbols-outlined text-[#f2ca50] text-base">help</span>
          <p className="text-[10px] text-[#f2ca50] font-bold uppercase tracking-widest">Agent demande</p>
        </div>
        <p className="text-sm text-white/80">{msg.text}</p>
      </div>
    )
  }
  if (msg.type === 'user') {
    return (
      <div className="flex justify-end my-2 group">
        <div className="bg-[#f2ca50]/10 backdrop-blur-sm border border-[#f2ca50]/20 rounded-xl rounded-tr-none px-4 py-3 max-w-[80%] relative">
          <p className="text-[9px] text-[#f2ca50]/50 uppercase tracking-widest mb-1">{msg.time}</p>
          <p className="text-sm text-white/90 pr-8">{msg.text}</p>
          <button
            onClick={() => onRerun(msg.text)}
            className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-1.5 rounded-lg bg-[#4ade80]/20 border border-[#4ade80]/40 hover:bg-[#4ade80]/30"
            title="Relancer cette recherche"
          >
            <span className="material-symbols-outlined text-[#4ade80] text-sm">replay</span>
          </button>
        </div>
      </div>
    )
  }
  // screenshots are intentionally NOT rendered in the conversation zone
  return null
}

export default function App() {
  const backendCandidates = [import.meta.env.VITE_BACKEND_URL, 'http://127.0.0.1:8000', 'http://127.0.0.1:8001'].filter(Boolean)
  const [auth, setAuth] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(AUTH_STORAGE_KEY) || 'null')
    } catch (_) {
      return null
    }
  })
  const [backendUrl, setBackendUrl] = useState('')
  const [backendAvailable, setBackendAvailable] = useState(false)
  const [command, setCommand] = useState('')
  const [agentStatus, setAgentStatus] = useState('idle')
  const [messages, setMessages] = useState([])
  const [sessions, setSessions] = useState([])
  const [activeSessionId, setActiveSessionId] = useState(() => localStorage.getItem(SESSION_STORAGE_KEY) || '')
  const [sessionsLoading, setSessionsLoading] = useState(false)
  const [currentUrl, setCurrentUrl] = useState('')
  const [currentTopic, setCurrentTopic] = useState('')
  const [sessionSummary, setSessionSummary] = useState('')
  const [activeTask, setActiveTask] = useState('')
  const [toolsUsed, setToolsUsed] = useState([])
  const [lastScreenshot, setLastScreenshot] = useState(null)
  const [screenshotHistory, setScreenshotHistory] = useState([])
  const [showScreenshotHistory, setShowScreenshotHistory] = useState(false)
  const [feedbackQuestion, setFeedbackQuestion] = useState('')
  const [showFeedback, setShowFeedback] = useState(false)
  const [showSafety, setShowSafety] = useState(false)
  const [safetyMsg, setSafetyMsg] = useState('')
  const [staleBrowser, setStaleBrowser] = useState(false)
  const [showBrowser, setShowBrowser] = useState(false)

  const scrollRef = useRef(null)
  const cmdRef = useRef(null)
  const controllerRef = useRef(null)
  const streamActiveRef = useRef(false)

  const authHeaders = auth?.token ? { Authorization: `Bearer ${auth.token}` } : {}

  const handleAuthenticated = (nextAuth) => {
    localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(nextAuth))
    setAuth(nextAuth)
    setBackendUrl(nextAuth.backendUrl)
    setBackendAvailable(true)
  }

  const handleLogout = () => {
    localStorage.removeItem(AUTH_STORAGE_KEY)
    localStorage.removeItem(SESSION_STORAGE_KEY)
    setAuth(null)
    setBackendUrl('')
    setBackendAvailable(false)
    setMessages([])
    setSessions([])
    setActiveSessionId('')
    setCommand('')
    setAgentStatus('idle')
  }

  const refreshSessions = async (preferredSessionId = activeSessionId) => {
    if (!auth?.token || !backendUrl) return
    setSessionsLoading(true)
    try {
      const response = await fetch(`${backendUrl}/sessions`, { headers: authHeaders })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const data = await response.json()
      // Deduplicate by task name — keep only the most recent per task
      const seen = new Set()
      const nextSessions = (data.sessions || []).filter(s => {
        const key = (s.task || '').trim().toLowerCase()
        if (seen.has(key)) return false
        seen.add(key)
        return true
      })
      setSessions(nextSessions)
      const nextActive = nextSessions.find(session => session.id === preferredSessionId)?.id || nextSessions[0]?.id || ''
      if (nextActive && nextActive !== activeSessionId) {
        await loadSession(nextActive, false)
      } else if (!nextActive) {
        setMessages([])
        setActiveSessionId('')
        localStorage.removeItem(SESSION_STORAGE_KEY)
      }
    } catch (_) {
      setSessions([])
    } finally {
      setSessionsLoading(false)
    }
  }

  const deleteSession = async (sessionId) => {
    if (!auth?.token || !backendUrl || !sessionId) return
    if (!confirm('Supprimer cette session et tous ses messages ?')) return
    try {
      const response = await fetch(`${backendUrl}/sessions/${sessionId}`, {
        method: 'DELETE',
        headers: authHeaders
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      
      // If deleted session was active, clear it
      if (sessionId === activeSessionId) {
        setMessages([])
        setActiveSessionId('')
        setLastScreenshot(null)
        setScreenshotHistory([])
        setCurrentUrl('')
        setCurrentTopic('')
        localStorage.removeItem(SESSION_STORAGE_KEY)
      }
      
      // Refresh sessions list
      await refreshSessions()
    } catch (err) {
      alert(`Erreur lors de la suppression: ${err.message}`)
    }
  }

  const loadSession = async (sessionId, refreshList = true) => {
    if (!auth?.token || !backendUrl || !sessionId) return
    try {
      const response = await fetch(`${backendUrl}/sessions/${sessionId}`, { headers: authHeaders })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const data = await response.json()
      setActiveSessionId(sessionId)
      localStorage.setItem(SESSION_STORAGE_KEY, sessionId)
      setActiveTask(data.session?.task || '')
      
      const screenshotsResponse = await fetch(`${backendUrl}/sessions/${sessionId}/screenshots`, { headers: authHeaders })
      if (screenshotsResponse.ok) {
        const screenshotsData = await screenshotsResponse.json()
        const screenshots = screenshotsData.screenshots || []
        setScreenshotHistory(screenshots)
        if (screenshots.length > 0) {
          setLastScreenshot(`data:image/png;base64,${screenshots[screenshots.length - 1].content}`)
        } else {
          setLastScreenshot(null)
        }
      } else {
        setScreenshotHistory([])
        setLastScreenshot(null)
      }
      
      if (data.session?.lastUrl) setCurrentUrl(data.session.lastUrl)
      else setCurrentUrl('')
      
      setMessages((data.messages || []).filter(m => m.messageType !== 'screenshot').map(mapStoredMessage))
      setAgentStatus('idle')
      setCurrentTopic(data.session?.task || '')
      setSessionSummary('')
      if (refreshList) await refreshSessions(sessionId)
    } catch (_) {
      setActiveSessionId('')
      localStorage.removeItem(SESSION_STORAGE_KEY)
    }
  }

  const handleNewSubject = async () => {
    if (agentStatus === 'executing') {
      return
    }
    setMessages([])
    setCurrentUrl('')
    setCurrentTopic('')
    setSessionSummary('')
    setActiveTask('')
    setToolsUsed([])
    setLastScreenshot(null)
    setScreenshotHistory([])
    setShowScreenshotHistory(false)
    setShowFeedback(false)
    setShowSafety(false)
    setAgentStatus('idle')
    setCommand('')
    setActiveSessionId('')
    localStorage.removeItem(SESSION_STORAGE_KEY)
  }

  useEffect(() => {
    if (auth?.backendUrl) {
      setBackendUrl(auth.backendUrl)
      setBackendAvailable(true)
      return
    }

    let canceled = false
    const ctrl = new AbortController();
    (async () => {
      for (const url of backendCandidates) {
        if (canceled) return
        try {
          const r = await fetch(`${url}/health`, { signal: ctrl.signal })
          if (r.ok && !canceled) { setBackendUrl(url); setBackendAvailable(true); return }
        } catch (_) {}
      }
      if (!canceled) setBackendAvailable(false)
    })()
    return () => { canceled = true; ctrl.abort() }
  }, [auth?.backendUrl])

  useEffect(() => {
    if (auth?.token && backendUrl) {
      refreshSessions(localStorage.getItem(SESSION_STORAGE_KEY) || activeSessionId)
    }
  }, [auth?.token, backendUrl])

  // Restore last session on load
  useEffect(() => {
    if (!backendUrl || !backendAvailable) return
    const token = localStorage.getItem('gsam_token')
    if (!token) return
    fetch(`${backendUrl}/session/restore`, {
      headers: { Authorization: `Bearer ${token}` }
    })
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (!data?.session) return
        const { lastUrl, lastScreenshot, task } = data.session
        if (lastUrl) setCurrentUrl(lastUrl)
        if (lastScreenshot) setLastScreenshot(`data:image/png;base64,${lastScreenshot}`)
        if (data.messages?.length > 0) {
          setMessages(data.messages.map(m => ({
            id: m.id,
            type: m.role === 'user' ? 'user' : m.messageType === 'result' ? 'result' : 'action',
            text: m.content,
            name: m.messageType === 'action' ? m.content : undefined,
            args: m.generatedBy || undefined,
            status: 'done',
            data: m.messageType === 'result' ? (() => { try { return JSON.parse(m.content) } catch { return { text: m.content } } })() : undefined,
            time: m.timestamp ? new Date(m.timestamp).toTimeString().slice(0, 8) : '',
          })))
        }
      })
      .catch(() => {})
  }, [backendUrl, backendAvailable])

  const addMsg = (msg) => setMessages(prev => [...prev, { ...msg, id: Date.now() + Math.random() }])

  const handleEvent = (event) => {
    switch (event.type) {
      case 'screenshot':
        setLastScreenshot(`data:image/png;base64,${event.data}`)
        // screenshots go to sidebar only, NOT to conversation
        break
      case 'url':
        setCurrentUrl(event.value || '')
        break
      case 'session':
        setCurrentTopic(event.data?.topic || '')
        setSessionSummary(event.data?.summary || '')
        break
      case 'session_started':
        if (event.session_id) {
          setActiveSessionId(event.session_id)
          localStorage.setItem(SESSION_STORAGE_KEY, event.session_id)
        }
        break
      case 'step':
        if (event.name) {
          const detectedTool = ['langchain', 'playwright', 'rpa', 'gemini', 'computer_use'].find(tool =>
            `${event.name} ${event.args || ''}`.toLowerCase().includes(tool)
          )
          if (detectedTool) setToolsUsed(prev => prev.includes(detectedTool) ? prev : [...prev, detectedTool])
        }
        setMessages(prev => {
          const exists = prev.find(m => m.type === 'action' && m.name === event.name && m.status !== 'done')
          if (exists) return prev.map(m => m.type === 'action' && m.name === event.name ? { ...m, status: event.status, args: event.args } : m)
          return [...prev, { id: Date.now(), type: 'action', name: event.name, args: event.args, status: event.status }]
        })
        break
      case 'result':
        addMsg({ type: 'result', data: event.data || {} })
        break
      case 'ask_user':
        setFeedbackQuestion(event.question || '')
        setAgentStatus('waiting')
        setShowFeedback(true)
        addMsg({ type: 'question', text: event.question || '' })
        break
      case 'safety':
        setSafetyMsg(event.explanation || '')
        setAgentStatus('waiting')
        setShowSafety(true)
        break
      case 'error':
        addMsg({ type: 'action', name: 'ERROR', args: event.message, status: 'error' })
        setAgentStatus('error')
        break
      case 'done':
        setAgentStatus(prev => prev === 'waiting' ? 'waiting' : 'complete')
        break
      default:
        break
    }
  }

  const parseSse = (chunk) => {
    const lines = chunk.replace(/\r\n/g, '\n').split('\n').filter(l => l.startsWith('data:'))
    const payload = lines.map(l => l.slice(5).trim()).join('')
    if (!payload) return
    try { handleEvent(JSON.parse(payload)) } catch (_) {}
  }

  const startAgent = async (task) => {
    if (streamActiveRef.current) {
      addMsg({ type: 'user', text: task, time: new Date().toTimeString().slice(0, 8) })
      setAgentStatus('executing')
      await fetch(`${backendUrl}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders },
        body: JSON.stringify({ message: task })
      }).catch(() => {})
      return
    }

    const continuingSessionId = activeSessionId
    if (!continuingSessionId) setMessages([])
    setCurrentUrl('')
    setCurrentTopic('')
    setSessionSummary('')
    setActiveTask(continuingSessionId ? activeTask : task)
    setToolsUsed([])
    setLastScreenshot(null)
    setAgentStatus('executing')
    addMsg({ type: 'user', text: task, time: new Date().toTimeString().slice(0, 8) })

    controllerRef.current = new AbortController()
    streamActiveRef.current = true
    try {
      const res = await fetch(`${backendUrl}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders },
        body: JSON.stringify({
          task,
          session_id: continuingSessionId || undefined,
          stale_browser: staleBrowser,
          skip_anti_bot: false,
          show_browser: showBrowser
        }),
        signal: controllerRef.current.signal
      })
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`)
      const reader = res.body.getReader()
      const dec = new TextDecoder()
      let buf = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        const parts = buf.split(/\n\n|\r\n\r\n/)
        buf = parts.pop()
        parts.forEach(p => p.trim() && parseSse(p))
      }
      if (buf.trim()) parseSse(buf)
    } catch (err) {
      if (err.name !== 'AbortError') {
        addMsg({ type: 'action', name: 'CONNECTION ERROR', args: err.message, status: 'error' })
        setAgentStatus('error')
      }
    } finally {
      streamActiveRef.current = false
      refreshSessions(localStorage.getItem(SESSION_STORAGE_KEY) || activeSessionId || continuingSessionId)
    }
  }

  const handleCommand = () => {
    if (!command.trim()) return
    if (!backendAvailable) return
    if (agentStatus === 'executing' && !streamActiveRef.current) return
    if (agentStatus === 'waiting') return
    startAgent(command.trim())
    setCommand('')
  }

  const handleFeedback = async (answer) => {
    setShowFeedback(false)
    setFeedbackQuestion('')
    setAgentStatus('executing')
    // Send raw answer — backend detects __PAGE_ACTION__: prefix itself
    const display = answer.startsWith('__PAGE_ACTION__:')
      ? answer.slice('__PAGE_ACTION__:'.length).trim()
      : answer
    addMsg({ type: 'user', text: display, time: new Date().toTimeString().slice(0, 8) })
    await fetch(`${backendUrl}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders },
      body: JSON.stringify({ message: answer })
    }).catch(() => {})
  }

  const handleSkipFeedback = async () => {
    setShowFeedback(false)
    setFeedbackQuestion('')
    setAgentStatus('idle')
    addMsg({ type: 'user', text: '[Question ignorée - retour à la conversation normale]', time: new Date().toTimeString().slice(0, 8) })
    // Abort the current agent task
    if (controllerRef.current) {
      controllerRef.current.abort()
    }
    await fetch(`${backendUrl}/abort`, {
      method: 'POST',
      headers: authHeaders
    }).catch(() => {})
  }

  const handleRerun = async (taskText) => {
    if (agentStatus === 'executing' || agentStatus === 'waiting') {
      return
    }
    // Rerun the same task - Playwright will use stale browser with history
    await startAgent(taskText)
  }

  const handleConfirm = async () => {
    setShowSafety(false)
    setAgentStatus('executing')
    await fetch(`${backendUrl}/confirm`, { method: 'POST', headers: authHeaders }).catch(() => {})
  }

  const handleReset = async () => {
    await handleNewSubject()
  }

  const statusColor = { idle: '#e5e2e1', executing: '#f2ca50', waiting: '#60a5fa', complete: '#4ade80', error: '#f87171' }[agentStatus] || '#e5e2e1'

  if (!auth?.token) {
    return <AuthPanel backendCandidates={backendCandidates} onAuthenticated={handleAuthenticated} />
  }

  return (
    <div className="flex h-screen w-full overflow-hidden text-white relative">

      {/* ── FULL-SCREEN BACKGROUND IMAGE ── */}
      <div
        className="fixed inset-0 z-0"
        style={{
          backgroundImage: `url(${ASSET_IMAGE})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
        }}
      />
      {/* Dark overlay so text stays readable */}
      <div className="fixed inset-0 z-0 bg-black/60" />

      <SessionSidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelect={loadSession}
        onNew={handleNewSubject}
        onDelete={deleteSession}
        loading={sessionsLoading}
        user={auth.user}
      />

      {/* ── MAIN PANEL ── */}
      <main className="flex-1 flex flex-col h-screen overflow-hidden relative z-10">

        {/* HEADER — glassmorphism */}
        <header className="flex items-center justify-between px-6 py-3 bg-black/30 backdrop-blur-md border-b border-white/10 shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full animate-pulse" style={{ background: statusColor, boxShadow: `0 0 8px ${statusColor}` }} />
            <span className="text-[10px] uppercase tracking-widest font-bold" style={{ color: statusColor }}>
              {agentStatus.toUpperCase()}
            </span>
          </div>

          {/* LOGO + TITLE */}
          <div className="flex items-center gap-3">
            <img
              src={ASSET_IMAGE}
              alt="GSAM"
              className="h-9 w-9 rounded-full object-cover border-2 border-[#f2ca50]/50"
              style={{ boxShadow: '0 0 12px rgba(242,202,80,0.4)' }}
            />
            <h1 className="text-[#f2ca50] font-black tracking-widest text-sm drop-shadow">GSAM | PRIVATE INTELLIGENCE</h1>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleReset}
              disabled={agentStatus === 'executing'}
              className="text-[9px] uppercase tracking-widest text-[#4ade80] border border-[#4ade80]/30 px-3 py-1.5 rounded hover:bg-[#4ade80]/10 backdrop-blur-sm disabled:opacity-30 disabled:hover:bg-transparent"
            >
              NOUVEAU SUJET
            </button>
            <button onClick={handleLogout} className="text-[9px] uppercase tracking-widest text-white/55 border border-white/15 px-3 py-1.5 rounded hover:bg-white/10 backdrop-blur-sm">
              LOGOUT
            </button>
          </div>
        </header>

        {!backendAvailable && (
          <div className="mx-4 mt-3 p-3 rounded border border-red-500/30 bg-red-500/10 backdrop-blur-sm text-[11px] text-red-400 uppercase tracking-widest">
            Backend indisponible — lance: uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
          </div>
        )}

        {/* MESSAGES AREA — fully transparent, background shows through */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6 space-y-1">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full gap-6">
              <img
                src={ASSET_IMAGE}
                alt="GSAM"
                className="w-28 h-28 rounded-full object-cover border-2 border-[#f2ca50]/40 opacity-80"
                style={{ boxShadow: '0 0 40px rgba(242,202,80,0.25)' }}
              />
              <p className="text-sm uppercase tracking-widest text-white/50">Entrez une mission ci-dessous</p>
            </div>
          )}
          {messages.map(msg => <AgentMessage key={msg.id} msg={msg} onRerun={handleRerun} />)}
        </div>

        {/* INPUT BAR — glassmorphism */}
        <div className="shrink-0 px-4 py-3 bg-black/30 backdrop-blur-md border-t border-white/10">
          <div className="flex items-center gap-2 bg-white/5 border border-white/10 rounded-lg px-3 py-2">
            <span className="material-symbols-outlined text-[#f2ca50]/60 text-lg">terminal</span>
            <input
              ref={cmdRef}
              value={command}
              onChange={e => setCommand(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); handleCommand() } }}
              disabled={agentStatus === 'executing' && !streamActiveRef.current}
              className="flex-1 bg-transparent text-sm focus:outline-none placeholder:text-white/30 disabled:opacity-40 text-white"
              placeholder={
                agentStatus === 'waiting' ? 'Répondez dans le modal ci-dessus...' :
                streamActiveRef.current ? 'Nouvelle instruction (ou "stop" pour arrêter)...' :
                'Entrez une mission...'
              }
            />
            <label className="flex items-center gap-1 text-[9px] uppercase tracking-widest text-white/40 cursor-pointer">
              <input type="checkbox" checked={staleBrowser} onChange={e => setStaleBrowser(e.target.checked)} className="w-3 h-3" />
              Stale
            </label>
            <label className="flex items-center gap-1 text-[9px] uppercase tracking-widest cursor-pointer" style={{ color: showBrowser ? '#60a5fa' : 'rgba(255,255,255,0.4)' }}>
              <input type="checkbox" checked={showBrowser} onChange={e => setShowBrowser(e.target.checked)} className="w-3 h-3" />
              <span className="material-symbols-outlined text-xs">desktop_windows</span>
              PC
            </label>
            <button
              onClick={handleCommand}
              disabled={(agentStatus === 'executing' && !streamActiveRef.current) || agentStatus === 'waiting' || !command.trim()}
              className="bg-gradient-to-r from-[#f2ca50] to-[#d4af37] text-[#3d2f00] font-black text-[9px] uppercase tracking-widest px-4 py-1.5 rounded disabled:opacity-30"
            >
              CMD
            </button>
          </div>
        </div>
      </main>

      {/* ── SIDEBAR — live screenshots ── */}
      <aside className="hidden xl:flex flex-col w-96 bg-black/30 backdrop-blur-md border-l border-white/10 relative z-10">
        <div className="p-3 border-b border-white/10 flex items-center gap-2">
          <span className="material-symbols-outlined text-[#f2ca50] text-sm">screenshot_monitor</span>
          <p className="text-[10px] uppercase tracking-widest text-[#f2ca50] font-bold">Live Browser</p>
          {currentUrl && (
            <a
              href={currentUrl}
              target="_blank"
              rel="noreferrer"
              title="Ouvrir dans le navigateur"
              className="ml-auto flex items-center gap-1 text-[9px] text-[#60a5fa]/70 hover:text-[#60a5fa] border border-[#60a5fa]/20 hover:border-[#60a5fa]/50 px-2 py-1 rounded transition-colors"
            >
              <span className="material-symbols-outlined text-xs">open_in_new</span>
              OUVRIR
            </a>
          )}
        </div>
        {currentUrl && (
          <div className="px-3 py-1.5 bg-black/20 border-b border-white/5">
            <p className="text-[9px] text-white/30 truncate">{currentUrl}</p>
          </div>
        )}
        <div className="flex-1 overflow-y-auto bg-white">
          {lastScreenshot ? (
            <div className="space-y-2 p-2">
              <img
                src={lastScreenshot}
                alt="browser"
                className="w-full h-auto block border-2 border-[#f2ca50]/30"
                style={{ imageRendering: 'crisp-edges' }}
              />
              {screenshotHistory.length > 1 && (
                <div className="space-y-2">
                  <button
                    onClick={() => setShowScreenshotHistory(!showScreenshotHistory)}
                    className="w-full text-[9px] uppercase tracking-widest text-black/60 hover:text-black border border-black/20 hover:border-black/40 px-2 py-1.5 rounded bg-white/80 hover:bg-white transition-colors flex items-center justify-center gap-1"
                  >
                    <span className="material-symbols-outlined text-xs">{showScreenshotHistory ? 'expand_less' : 'expand_more'}</span>
                    Historique ({screenshotHistory.length - 1})
                  </button>
                  {showScreenshotHistory && screenshotHistory.slice(0, -1).reverse().map((ss, idx) => (
                    <div key={idx} className="border border-white/20 rounded overflow-hidden">
                      <img
                        src={`data:image/png;base64,${ss.content}`}
                        alt={`screenshot ${idx}`}
                        className="w-full h-auto block cursor-pointer hover:opacity-80"
                        style={{ imageRendering: 'crisp-edges' }}
                        onClick={() => setLastScreenshot(`data:image/png;base64,${ss.content}`)}
                      />
                      {ss.pageUrl && (
                        <div className="bg-black/80 px-2 py-1">
                          <p className="text-[8px] text-white/60 truncate">{ss.pageUrl}</p>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-full gap-3 bg-black/20">
              <span className="material-symbols-outlined text-white/20 text-4xl">travel_explore</span>
              <p className="text-white/20 text-[10px] uppercase tracking-widest">En attente...</p>
            </div>
          )}
        </div>
      </aside>

      {/* ── FEEDBACK MODAL ── */}
      {showFeedback && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 w-full max-w-lg px-4">
          <div className="bg-black/25 backdrop-blur-lg border border-[#f2ca50]/30 rounded-2xl p-5 shadow-2xl">
            <p className="text-[11px] uppercase tracking-widest font-bold text-[#f2ca50] mb-3">
              {feedbackQuestion.replace(/\s*\(options:[^)]+\)/i, '')}
            </p>
            <ActionableFeedbackInput question={feedbackQuestion} onSubmit={handleFeedback} onSkip={handleSkipFeedback} />
          </div>
        </div>
      )}

      {/* ── SAFETY MODAL ── */}
      {showSafety && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-black/60 backdrop-blur-xl border border-[#f2ca50]/20 rounded-2xl p-6 w-full max-w-md shadow-2xl">
            <span className="material-symbols-outlined text-[#f2ca50] text-4xl block text-center mb-3">gpp_maybe</span>
            <p className="text-[11px] uppercase tracking-widest font-bold text-[#f2ca50] text-center mb-3">Confirmation requise</p>
            <p className="text-sm text-white/70 text-center mb-6">{safetyMsg}</p>
            <button onClick={handleConfirm} className="w-full py-2.5 text-[10px] uppercase tracking-widest font-bold bg-gradient-to-r from-[#f2ca50] to-[#d4af37] text-[#3d2f00] rounded-lg">CONFIRM</button>
          </div>
        </div>
      )}
    </div>
  )
}
