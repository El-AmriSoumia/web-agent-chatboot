import { useEffect, useRef, useState } from 'react'

const ASSET_IMAGE = '/gsam-bg.png'
const PAGE_ACTION_PREFIX = '__PAGE_ACTION__:'

function FeedbackInput({ question, onSubmit, onAbort }) {
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
        <button
          onClick={onAbort}
          className="w-full py-2 text-[10px] uppercase tracking-widest text-[#ffb4ab] border border-[#ffb4ab]/20 rounded hover:bg-[#ffb4ab]/10"
        >
          ABORT
        </button>
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
        <button onClick={onAbort} className="flex-1 py-2 text-[10px] uppercase tracking-widest text-[#ffb4ab] border border-[#ffb4ab]/20 rounded hover:bg-[#ffb4ab]/10">ABORT</button>
        <button onClick={() => text.trim() && onSubmit(text.trim())}
          className="flex-1 py-2 text-[10px] uppercase tracking-widest font-bold bg-gradient-to-r from-[#f2ca50] to-[#d4af37] text-[#3d2f00] rounded">SEND</button>
      </div>
    </div>
  )
}

function ActionableFeedbackInput({ question, onSubmit, onAbort }) {
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
          <button onClick={onAbort} className="flex-1 py-2 text-[10px] uppercase tracking-widest text-[#ffb4ab] border border-[#ffb4ab]/20 rounded hover:bg-[#ffb4ab]/10">ABORT</button>
          <button
            onClick={() => { setMode('redirect'); setText('') }}
            className="flex-1 py-2 text-[10px] uppercase tracking-widest text-[#60a5fa] border border-[#60a5fa]/25 rounded hover:bg-[#60a5fa]/10"
          >
            AUTRE ACTION
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
        <button onClick={onAbort} className="flex-1 py-2 text-[10px] uppercase tracking-widest text-[#ffb4ab] border border-[#ffb4ab]/20 rounded hover:bg-[#ffb4ab]/10">ABORT</button>
        <button
          onClick={() => { setMode(mode === 'redirect' ? 'answer' : 'redirect'); setText('') }}
          className="flex-1 py-2 text-[10px] uppercase tracking-widest text-[#60a5fa] border border-[#60a5fa]/25 rounded hover:bg-[#60a5fa]/10"
        >
          {mode === 'redirect' ? 'RÉPONSE NORMALE' : 'AUTRE ACTION'}
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

function AgentMessage({ msg }) {
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
      <div className="flex justify-end my-2">
        <div className="bg-[#f2ca50]/10 backdrop-blur-sm border border-[#f2ca50]/20 rounded-xl rounded-tr-none px-4 py-3 max-w-[80%]">
          <p className="text-[9px] text-[#f2ca50]/50 uppercase tracking-widest mb-1">{msg.time}</p>
          <p className="text-sm text-white/90">{msg.text}</p>
        </div>
      </div>
    )
  }
  // screenshots are intentionally NOT rendered in the conversation zone
  if (msg.type === 'abort') {
    return <p className="text-center text-[#ffb4ab] text-[11px] uppercase tracking-widest py-3">⊗ Mission abortée</p>
  }
  return null
}

export default function App() {
  const backendCandidates = [import.meta.env.VITE_BACKEND_URL, 'http://127.0.0.1:8000', 'http://127.0.0.1:8001'].filter(Boolean)
  const [backendUrl, setBackendUrl] = useState('')
  const [backendAvailable, setBackendAvailable] = useState(false)
  const [command, setCommand] = useState('')
  const [agentStatus, setAgentStatus] = useState('idle')
  const [messages, setMessages] = useState([])
  const [currentUrl, setCurrentUrl] = useState('')
  const [currentTopic, setCurrentTopic] = useState('')
  const [sessionSummary, setSessionSummary] = useState('')
  const [lastScreenshot, setLastScreenshot] = useState(null)
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

  useEffect(() => {
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
  }, [])

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [messages])

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
      case 'step':
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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: task })
      }).catch(() => {})
      return
    }

    setMessages([])
    setCurrentUrl('')
    setCurrentTopic('')
    setSessionSummary('')
    setLastScreenshot(null)
    setAgentStatus('executing')
    addMsg({ type: 'user', text: task, time: new Date().toTimeString().slice(0, 8) })

    controllerRef.current = new AbortController()
    streamActiveRef.current = true
    try {
      const res = await fetch(`${backendUrl}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task, stale_browser: staleBrowser, skip_anti_bot: true, show_browser: showBrowser }),
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
    addMsg({ type: 'user', text: answer, time: new Date().toTimeString().slice(0, 8) })
    await fetch(`${backendUrl}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: answer })
    }).catch(() => {})
  }

  const handleConfirm = async () => {
    setShowSafety(false)
    setAgentStatus('executing')
    await fetch(`${backendUrl}/confirm`, { method: 'POST' }).catch(() => {})
  }

  const handleReset = async () => {
    controllerRef.current?.abort()
    streamActiveRef.current = false
    await fetch(`${backendUrl}/reset`, { method: 'POST' }).catch(() => {})
    setMessages([])
    setCurrentUrl('')
    setCurrentTopic('')
    setSessionSummary('')
    setLastScreenshot(null)
    setShowFeedback(false)
    setShowSafety(false)
    setAgentStatus('idle')
    setCommand('')
  }

  const handleAbort = async () => {
    controllerRef.current?.abort()
    streamActiveRef.current = false
    await fetch(`${backendUrl}/abort`, { method: 'POST' }).catch(() => {})
    setShowFeedback(false)
    setShowSafety(false)
    setAgentStatus('idle')
    addMsg({ type: 'abort' })
  }

  const statusColor = { idle: '#e5e2e1', executing: '#f2ca50', waiting: '#60a5fa', complete: '#4ade80', error: '#f87171' }[agentStatus] || '#e5e2e1'

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

      {/* ── MAIN PANEL ── */}
      <main className="flex-1 flex flex-col h-screen overflow-hidden relative z-10">

        {/* HEADER — glassmorphism */}
        <header className="flex items-center justify-between px-6 py-3 bg-black/30 backdrop-blur-md border-b border-white/10 shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full animate-pulse" style={{ background: statusColor, boxShadow: `0 0 8px ${statusColor}` }} />
            <span className="text-[10px] uppercase tracking-widest font-bold" style={{ color: statusColor }}>
              {agentStatus.toUpperCase()}
            </span>
            {currentTopic && (
              <span
                className="max-w-[18rem] truncate text-[9px] uppercase tracking-[0.25em] text-white/55 border border-white/10 rounded-full px-3 py-1"
                title={sessionSummary || currentTopic}
              >
                {currentTopic}
              </span>
            )}
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
            <button onClick={handleReset} className="text-[9px] uppercase tracking-widest text-[#4ade80] border border-[#4ade80]/30 px-3 py-1.5 rounded hover:bg-[#4ade80]/10 backdrop-blur-sm">
              NOUVEAU SUJET
            </button>
            <button onClick={handleAbort} className="text-[9px] uppercase tracking-widest text-[#ffb4ab] border border-[#ffb4ab]/30 px-3 py-1.5 rounded hover:bg-[#ffb4ab]/10 backdrop-blur-sm">
              ABORT
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
          {messages.map(msg => <AgentMessage key={msg.id} msg={msg} />)}
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
            <img
              src={lastScreenshot}
              alt="browser"
              className="w-full h-auto block"
              style={{ imageRendering: 'crisp-edges' }}
            />
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
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-black/60 backdrop-blur-xl border border-[#f2ca50]/20 rounded-2xl p-6 w-full max-w-lg shadow-2xl">
            <div className="flex items-center gap-2 mb-4">
              <span className="material-symbols-outlined text-[#f2ca50]">help</span>
              <p className="text-[11px] uppercase tracking-widest font-bold text-[#f2ca50]">L'agent demande</p>
            </div>
            <p className="text-sm text-white/80 mb-4">{feedbackQuestion.replace(/\s*\(options:[^)]+\)/i, '')}</p>
            <ActionableFeedbackInput question={feedbackQuestion} onSubmit={handleFeedback} onAbort={handleAbort} />
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
            <div className="flex gap-3">
              <button onClick={handleAbort} className="flex-1 py-2.5 text-[10px] uppercase tracking-widest text-[#ffb4ab] border border-[#ffb4ab]/20 rounded-lg hover:bg-[#ffb4ab]/10">ABORT</button>
              <button onClick={handleConfirm} className="flex-1 py-2.5 text-[10px] uppercase tracking-widest font-bold bg-gradient-to-r from-[#f2ca50] to-[#d4af37] text-[#3d2f00] rounded-lg">CONFIRM</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
