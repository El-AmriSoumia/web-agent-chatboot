import { useEffect, useRef, useState } from 'react'

const ASSET_IMAGE = '/gsam-bg.png'

function FeedbackInput({ question, onSubmit, onAbort }) {
  const [values, setValues] = useState({})
  const [text, setText] = useState('')

  const fieldMatch = question?.match(/(?:for[:\s]+)(.+)/i)
  const fields = fieldMatch
    ? fieldMatch[1].split(/,/).map(f => f.trim()).filter(Boolean)
    : null

  if (fields && fields.length > 1) {
    return (
      <div className="space-y-3">
        {fields.map(f => (
          <div key={f}>
            <label className="text-[9px] uppercase tracking-widest text-[#f2ca50]/70 mb-1 block">{f}</label>
            <input
              autoFocus={fields[0] === f}
              value={values[f] || ''}
              onChange={e => setValues(p => ({ ...p, [f]: e.target.value }))}
              onKeyDown={e => { if (e.key === 'Enter') document.getElementById('feedback-send')?.click() }}
              className="w-full bg-[#111] border border-[#4d4635]/40 rounded px-3 py-2 text-sm text-[#e5e2e1] focus:outline-none focus:border-[#f2ca50]/60"
              placeholder={`${f}...`}
            />
          </div>
        ))}
        <div className="flex gap-2 pt-1">
          <button onClick={onAbort} className="flex-1 py-2 text-[10px] uppercase tracking-widest text-[#ffb4ab] border border-[#ffb4ab]/20 rounded hover:bg-[#ffb4ab]/10">ABORT</button>
          <button id="feedback-send" onClick={() => onSubmit(fields.map(f => `${f}: ${values[f] || ''}`).join(', '))}
            className="flex-1 py-2 text-[10px] uppercase tracking-widest font-bold bg-gradient-to-r from-[#f2ca50] to-[#d4af37] text-[#3d2f00] rounded">SEND</button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <textarea
        autoFocus
        rows={3}
        value={text}
        onChange={e => setText(e.target.value)}
        onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && text.trim()) { e.preventDefault(); onSubmit(text.trim()) } }}
        className="w-full bg-[#111] border border-[#4d4635]/40 rounded px-3 py-2 text-sm text-[#e5e2e1] focus:outline-none focus:border-[#f2ca50]/60 resize-none"
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

function AgentMessage({ msg }) {
  if (msg.type === 'action') {
    return (
      <div className="flex items-start gap-3 py-2">
        <span className="material-symbols-outlined text-[#f2ca50] text-base mt-0.5">
          {msg.status === 'done' ? 'check_circle' : msg.status === 'running' ? 'pending' : 'radio_button_unchecked'}
        </span>
        <div>
          <p className="text-[11px] uppercase tracking-widest font-bold text-[#f2ca50]">{msg.name}</p>
          {msg.args && <p className="text-[11px] text-[#e5e2e1]/50 mt-0.5">{msg.args}</p>}
        </div>
      </div>
    )
  }
  if (msg.type === 'result') {
    return (
      <div className="bg-[#0e0e0e] border border-[#f2ca50]/20 rounded-xl p-5 my-3">
        <p className="text-[10px] text-[#f2ca50] font-bold uppercase tracking-widest mb-3">✓ Résultat</p>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(msg.data).map(([k, v]) => (
            <div key={k} className="bg-[#1a1a1a] rounded p-3">
              <p className="text-[9px] text-[#e5e2e1]/40 uppercase tracking-widest mb-1">{k.replace(/_/g, ' ')}</p>
              <p className="text-sm font-bold text-[#f2ca50] break-all">{String(v)}</p>
            </div>
          ))}
        </div>
      </div>
    )
  }
  if (msg.type === 'question') {
    return (
      <div className="bg-[#1a1400] border border-[#f2ca50]/30 rounded-xl p-4 my-3">
        <div className="flex items-center gap-2 mb-2">
          <span className="material-symbols-outlined text-[#f2ca50] text-base">help</span>
          <p className="text-[10px] text-[#f2ca50] font-bold uppercase tracking-widest">Agent demande</p>
        </div>
        <p className="text-sm text-[#e5e2e1]/80">{msg.text}</p>
      </div>
    )
  }
  if (msg.type === 'user') {
    return (
      <div className="flex justify-end my-2">
        <div className="bg-[#1c1b1b] border border-[#4d4635]/20 rounded-xl rounded-tr-none px-4 py-3 max-w-[80%]">
          <p className="text-[9px] text-[#f2ca50]/50 uppercase tracking-widest mb-1">{msg.time}</p>
          <p className="text-sm text-[#e5e2e1]/90">{msg.text}</p>
        </div>
      </div>
    )
  }
  if (msg.type === 'screenshot') {
    return (
      <div className="my-3 rounded-xl overflow-hidden border border-[#4d4635]/20">
        <img src={msg.src} alt="screenshot" className="w-full h-auto block" />
        {msg.url && <p className="text-[9px] text-[#e5e2e1]/30 px-3 py-1 bg-[#0e0e0e] truncate">{msg.url}</p>}
      </div>
    )
  }
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
  const [agentStatus, setAgentStatus] = useState('idle') // idle | executing | waiting | complete | error
  const [messages, setMessages] = useState([])
  const [currentUrl, setCurrentUrl] = useState('')
  const [lastScreenshot, setLastScreenshot] = useState(null)
  const [feedbackQuestion, setFeedbackQuestion] = useState('')
  const [showFeedback, setShowFeedback] = useState(false)
  const [showSafety, setShowSafety] = useState(false)
  const [safetyMsg, setSafetyMsg] = useState('')
  const [staleBrowser, setStaleBrowser] = useState(false)

  const scrollRef = useRef(null)
  const cmdRef = useRef(null)
  const controllerRef = useRef(null)

  // probe backend
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
        if (event.data) addMsg({ type: 'screenshot', src: `data:image/png;base64,${event.data}`, url: currentUrl })
        break
      case 'url':
        setCurrentUrl(event.value || '')
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
        setAgentStatus('complete')
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
    setMessages([])
    setCurrentUrl('')
    setLastScreenshot(null)
    setAgentStatus('executing')
    addMsg({ type: 'user', text: task, time: new Date().toTimeString().slice(0, 8) })

    controllerRef.current = new AbortController()
    try {
      const res = await fetch(`${backendUrl}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task, stale_browser: staleBrowser, skip_anti_bot: true }),
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
    }
  }

  const handleCommand = () => {
    if (!command.trim() || agentStatus === 'executing' || agentStatus === 'waiting') return
    if (!backendAvailable) return
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

  const handleAbort = async () => {
    controllerRef.current?.abort()
    await fetch(`${backendUrl}/abort`, { method: 'POST' }).catch(() => {})
    setShowFeedback(false)
    setShowSafety(false)
    setAgentStatus('idle')
    addMsg({ type: 'abort' })
  }

  const statusColor = { idle: '#e5e2e1', executing: '#f2ca50', waiting: '#60a5fa', complete: '#4ade80', error: '#f87171' }[agentStatus] || '#e5e2e1'

  return (
    <div className="flex h-screen w-full bg-[#0a0a0a] text-[#e5e2e1] overflow-hidden">
      <div className="fixed inset-0 z-[-2] bg-cover bg-center opacity-10" style={{ backgroundImage: `url(${ASSET_IMAGE})` }} />

      {/* MAIN PANEL */}
      <main className="flex-1 flex flex-col h-screen overflow-hidden">

        {/* HEADER */}
        <header className="flex items-center justify-between px-6 py-3 bg-[#111] border-b border-[#4d4635]/20 shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full animate-pulse" style={{ background: statusColor, boxShadow: `0 0 8px ${statusColor}` }} />
            <span className="text-[10px] uppercase tracking-widest font-bold" style={{ color: statusColor }}>
              {agentStatus.toUpperCase()}
            </span>
          </div>
          <h1 className="text-[#f2ca50] font-black tracking-widest text-sm">GSAM | PRIVATE INTELLIGENCE</h1>
          <button onClick={handleAbort} className="text-[9px] uppercase tracking-widest text-[#ffb4ab] border border-[#ffb4ab]/20 px-3 py-1.5 rounded hover:bg-[#ffb4ab]/10">
            ABORT
          </button>
        </header>

        {!backendAvailable && (
          <div className="mx-4 mt-3 p-3 rounded border border-red-500/30 bg-red-500/10 text-[11px] text-red-400 uppercase tracking-widest">
            Backend indisponible — lance: uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
          </div>
        )}

        {/* MESSAGES AREA */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6 space-y-1">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full gap-4 opacity-30">
              <span className="material-symbols-outlined text-6xl text-[#f2ca50]">terminal</span>
              <p className="text-sm uppercase tracking-widest">Entrez une mission ci-dessous</p>
            </div>
          )}
          {messages.map(msg => <AgentMessage key={msg.id} msg={msg} />)}
        </div>

        {/* INPUT BAR */}
        <div className="shrink-0 px-4 py-3 bg-[#111] border-t border-[#4d4635]/20">
          <div className="flex items-center gap-2 bg-[#1a1a1a] border border-[#4d4635]/30 rounded-lg px-3 py-2">
            <span className="material-symbols-outlined text-[#f2ca50]/40 text-lg">terminal</span>
            <input
              ref={cmdRef}
              value={command}
              onChange={e => setCommand(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); handleCommand() } }}
              disabled={agentStatus === 'executing' || agentStatus === 'waiting'}
              className="flex-1 bg-transparent text-sm focus:outline-none placeholder:text-[#e5e2e1]/20 disabled:opacity-40"
              placeholder={agentStatus === 'waiting' ? 'En attente de votre réponse...' : 'Entrez une mission...'}
            />
            <label className="flex items-center gap-1 text-[9px] uppercase tracking-widest text-[#e5e2e1]/40 cursor-pointer">
              <input type="checkbox" checked={staleBrowser} onChange={e => setStaleBrowser(e.target.checked)} className="w-3 h-3" />
              Stale
            </label>
            <button onClick={handleCommand} disabled={agentStatus === 'executing' || agentStatus === 'waiting' || !command.trim()}
              className="bg-gradient-to-r from-[#f2ca50] to-[#d4af37] text-[#3d2f00] font-black text-[9px] uppercase tracking-widest px-4 py-1.5 rounded disabled:opacity-30">
              CMD
            </button>
          </div>
        </div>
      </main>

      {/* SIDEBAR — screenshot */}
      <aside className="hidden xl:flex flex-col w-80 bg-[#0e0e0e] border-l border-[#4d4635]/20">
        <div className="p-3 border-b border-[#4d4635]/20">
          <p className="text-[10px] uppercase tracking-widest text-[#f2ca50] font-bold">Live Browser</p>
          {currentUrl && <p className="text-[9px] text-[#e5e2e1]/30 truncate mt-1">{currentUrl}</p>}
        </div>
        <div className="flex-1 bg-black overflow-hidden">
          {lastScreenshot
            ? <img src={lastScreenshot} alt="browser" className="w-full h-auto" />
            : <div className="flex items-center justify-center h-48 text-[#e5e2e1]/20 text-[10px] uppercase tracking-widest">En attente...</div>
          }
        </div>
      </aside>

      {/* FEEDBACK MODAL */}
      {showFeedback && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#1a1a1a] border border-[#f2ca50]/20 rounded-2xl p-6 w-full max-w-lg shadow-2xl">
            <div className="flex items-center gap-2 mb-4">
              <span className="material-symbols-outlined text-[#f2ca50]">help</span>
              <p className="text-[11px] uppercase tracking-widest font-bold text-[#f2ca50]">L'agent demande</p>
            </div>
            <p className="text-sm text-[#e5e2e1]/80 mb-5">{feedbackQuestion}</p>
            <FeedbackInput question={feedbackQuestion} onSubmit={handleFeedback} onAbort={handleAbort} />
          </div>
        </div>
      )}

      {/* SAFETY MODAL */}
      {showSafety && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#1a1a1a] border border-[#f2ca50]/20 rounded-2xl p-6 w-full max-w-md shadow-2xl">
            <span className="material-symbols-outlined text-[#f2ca50] text-4xl block text-center mb-3">gpp_maybe</span>
            <p className="text-[11px] uppercase tracking-widest font-bold text-[#f2ca50] text-center mb-3">Confirmation requise</p>
            <p className="text-sm text-[#e5e2e1]/70 text-center mb-6">{safetyMsg}</p>
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
