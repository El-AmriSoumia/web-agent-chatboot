import { useEffect, useRef, useState } from 'react'

const ASSET_IMAGE = '/gsam-bg.png'
const ASSET_FALLBACK = 'https://lh3.googleusercontent.com/aida-public/AB6AXuAUIM0D8nnGRwv_7GGid6UFv98WwpaiRop_Gtpp67YfqZkBz9sAi4-3BFd0MhBxBlbSXuHqY9mJaO599DJk3_kSMBpP_gMnMiTna-be9IeMdlUvo63hnKy__gXNTheC4550o0JMQvqCdJ9gz5_MnI1EOXhSlgfYdVD1tXzcDDpdXLT7mu4zXj_RLN0pwrYEhk87oONWHhRC_bTQqMCHVfFnEmqUl4zXteVu14wnoRzs2DI3Fcs5X8MyeoJk40Hn9_xz5U3tge0TPrC0'

const INITIAL_LOGS = [
  { time: '14:21:44', text: 'NAVIGATING TO ROOT...', active: false },
  { time: '14:22:01', text: 'RESOLVING SSL HANDSHAKE...', active: false },
  { time: '14:22:12', text: 'EXTRACTING NODE METADATA...', active: true },
  { time: '--:--:--', text: 'AWAITING NEXT ACTION', active: false, faded: true }
]

function App() {
  const backendCandidates = [import.meta.env.VITE_BACKEND_URL, 'http://127.0.0.1:8001', 'http://127.0.0.1:8000'].filter(Boolean)
  const [backendUrl, setBackendUrl] = useState('')
  const [userName, setUserName] = useState('Analyst')
  const [command, setCommand] = useState('')
  const [logs, setLogs] = useState(INITIAL_LOGS)
  const [steps, setSteps] = useState([])
  const [agentStatus, setAgentStatus] = useState('idle')
  const [currentTurn, setCurrentTurn] = useState(0)
  const [screenshot, setScreenshot] = useState(null)
  const [currentUrl, setCurrentUrl] = useState('')
  const [showModal, setShowModal] = useState(false)
  const [safetyMessage, setSafetyMessage] = useState('')
  const [toastVisible, setToastVisible] = useState(false)
  const [started, setStarted] = useState(false)
  const [chatEvents, setChatEvents] = useState([])
  const [resultData, setResultData] = useState(null)
  const [backendAvailable, setBackendAvailable] = useState(false)
  const [staleBrowser, setStaleBrowser] = useState(false)
  const [skipAntiBot, setSkipAntiBot] = useState(true)
  const backendControllerRef = useRef(null)
  const chatCanvasRef = useRef(null)
  const timersRef = useRef([])
  const abortedRef = useRef(false)
  const resumeRef = useRef(null)
  const commandInputRef = useRef(null)

  useEffect(() => {
    return () => {
      timersRef.current.forEach((id) => clearTimeout(id))
      timersRef.current = []
    }
  }, [])

  useEffect(() => {
    if (!toastVisible) return undefined
    const timeout = window.setTimeout(() => setToastVisible(false), 5000)
    return () => window.clearTimeout(timeout)
  }, [toastVisible])

  useEffect(() => {
    let canceled = false
    const controller = new AbortController()

    const probeBackend = async () => {
      for (const url of backendCandidates) {
        if (canceled) return
        try {
          const response = await fetch(`${url}/health`, { signal: controller.signal })
          if (response.ok) {
            if (!canceled) {
              setBackendUrl(url)
              setBackendAvailable(true)
              console.log('FRONTEND: Backend selected:', url)
            }
            return
          }
        } catch (_err) {
          console.log('FRONTEND: Backend probe failed for:', url, _err.message)
        }
      }
      if (!canceled) setBackendAvailable(false)
    }

    probeBackend()

    return () => {
      canceled = true
      controller.abort()
    }
  }, [])

  useEffect(() => {
    if (chatCanvasRef.current) {
      chatCanvasRef.current.scrollTop = chatCanvasRef.current.scrollHeight
    }
  }, [chatEvents])

  const addLog = (text, active = false) => {
    setLogs((prev) => {
      const next = [{ time: new Date().toTimeString().slice(0, 8), text, active }, ...prev]
      return next.slice(0, 6)
    })
  }

  const backendCleanup = () => {
    if (backendControllerRef.current) {
      backendControllerRef.current.abort()
      backendControllerRef.current = null
    }
  }

  const updateStep = ({ name, args = '', status }) => {
    setSteps((prev) => {
      const existing = prev.find((step) => step.name === name)
      if (existing) {
        return prev.map((step) => (step.name === name ? { ...step, args, status } : step))
      }
      return [...prev, { name, args, status }]
    })
    if (status === 'done') {
      setCurrentTurn((prev) => Math.min(prev + 1, 15))
    }
  }

  const handleBackendEvent = (event) => {
    switch (event.type) {
      case 'log':
        addLog(event.message || 'Log event', false)
        break
      case 'result':
        setResultData(event.data || {})
        setChatEvents((prev) => [...prev, { type: 'result', data: event.data || {} }])
        addLog('RESULT RECEIVED', false)
        break
      case 'url':
        if (event.value) setCurrentUrl(event.value)
        break
      case 'screenshot':
        if (event.data) setScreenshot(`data:image/png;base64,${event.data}`)
        break
      case 'step':
        updateStep({ name: event.name || 'step', args: event.args || '', status: event.status || 'pending' })
        break
      case 'safety':
        setSafetyMessage(event.explanation || 'The agent is requesting approval for a sensitive action.')
        setAgentStatus('waiting_confirmation')
        setShowModal(true)
        addLog(`SAFETY: ${event.explanation || 'Awaiting confirmation'}`, false)
        break
      case 'nlp': {
        const subtasks = Array.isArray(event.subtasks)
          ? event.subtasks
          : event.subtasks
            ? [event.subtasks]
            : []
        setChatEvents((prev) => [
          ...prev,
          {
            type: 'nlp',
            intent: event.intent || 'UNKNOWN',
            entity: event.entity || 'UNKNOWN',
            subtask: subtasks.length > 0 ? subtasks[subtasks.length - 1] : 'UNKNOWN',
            subtasks
          }
        ])
        break
      }
      case 'error':
        addLog(`ERROR: ${event.message || 'Unknown error'}`, false)
        setAgentStatus('error')
        break
      case 'done':
        setAgentStatus('complete')
        setToastVisible(true)
        break
      default:
        addLog(`EVENT: ${JSON.stringify(event)}`, false)
        break
    }
  }

  const parseSseChunk = (chunk, onEvent) => {
    const normalized = chunk.replace(/\r\n/g, '\n').replace(/\r/g, '\n')
    const lines = normalized.split('\n')
    const dataLines = lines.filter((line) => line.startsWith('data:'))
    if (!dataLines.length) return
    const payload = dataLines.map((line) => line.slice(5).trim()).join('')
    if (!payload) return
    try {
      const event = JSON.parse(payload)
      onEvent(event)
    } catch (err) {
      addLog(`SSE parse error: ${err.message}`, false)
      addLog(`SSE payload: ${payload.slice(0, 200)}`, false)
    }
  }

  const startBackendAgent = async (task) => {
    resetAgent()
    setStarted(true)
    setAgentStatus('executing')
    setSteps([])
    setCurrentTurn(0)
    setCurrentUrl('')
    setScreenshot(null)
    setShowModal(false)
    addLog('Backend task started.', true)

    backendControllerRef.current = new AbortController()
    try {
      const response = await fetch(`${backendUrl}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task, stale_browser: staleBrowser, skip_anti_bot: skipAntiBot }),
        signal: backendControllerRef.current.signal
      })

      if (!response.ok || !response.body) {
        throw new Error(`Backend request failed: ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split(/\n\n|\r\n\r\n/)
        buffer = parts.pop()

        for (const part of parts) {
          if (!part.trim()) continue
          parseSseChunk(part, handleBackendEvent)
        }
      }

      if (buffer.trim()) {
        parseSseChunk(buffer, handleBackendEvent)
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        addLog('Backend request aborted.', false)
      } else {
        addLog(`Erreur connexion backend: ${err.message}. Vérifiez que le serveur tourne sur ${backendUrl}.`, false)
        setAgentStatus('error')
      }
    } finally {
      backendCleanup()
    }
  }

  const handleCommand = () => {
    if (!command.trim() || agentStatus === 'executing') return
    if (!backendAvailable) {
      addLog('Backend indisponible. Démarre le serveur backend et réessaie.', false)
      return
    }
    appendUserMessage(command.trim())
    setCommand('')
    if (commandInputRef.current) commandInputRef.current.blur()
    startBackendAgent(command.trim())
  }

  const handleAbort = async () => {
    abortedRef.current = true
    if (backendControllerRef.current) {
      backendControllerRef.current.abort()
      try {
        await fetch(`${backendUrl}/abort`, { method: 'POST' })
      } catch (_) {
        // ignore abort errors
      }
      backendControllerRef.current = null
    }
    resetAgent()
    setAgentStatus('aborted')
    addLog('Agent loop terminated by operator.', false)
    setChatEvents((prev) => [...prev, { type: 'abort' }])
  }

  const appendUserMessage = (text) => {
    setChatEvents((prev) => [...prev, { type: 'user', text, time: new Date().toTimeString().slice(0, 8) }])
  }

  const resetAgent = () => {
    timersRef.current.forEach((id) => clearTimeout(id))
    timersRef.current = []
    abortedRef.current = false
    resumeRef.current = null
  }


  const handleConfirm = async () => {
    try {
      await fetch(`${backendUrl}/confirm`, { method: 'POST' })
    } catch (_) {
      // Ignore confirmation delivery failures; continue locally if needed.
    }
    setShowModal(false)
    setSafetyMessage('')
    setAgentStatus('executing')
    if (resumeRef.current) resumeRef.current()
  }

  const exportCsv = () => {
    const data = resultData || { Endpoints: 1242, Vulnerabilities: 12, Uptime: '99.8%' }
    const rows = Object.entries(data)
      .map(([k, v]) => `${k},${v}`)
      .join('\n')
    const csv = `Metric,Value\n${rows}`
    const blob = new Blob([csv], { type: 'text/csv' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'gsam-report.csv'
    a.click()
  }

  const exportPdf = () => {
    window.print()
  }

  const progressWidth = `${(currentTurn / 15) * 100}%`

  return (
    <div className="app-root relative text-on-surface overflow-hidden min-h-screen">
      <div className="fixed inset-0 z-[-2] bg-cover bg-center" style={{ backgroundImage: `url(${ASSET_IMAGE}), url(${ASSET_FALLBACK})` }} />
      <div className="fixed inset-0 z-[-1] bg-black/40 pointer-events-none" />
      <div className="flex h-screen w-full overflow-hidden">

        <main className="flex-1 flex flex-col relative h-screen bg-transparent overflow-hidden">
          {!backendAvailable && (
            <div className="absolute inset-x-0 top-0 z-50 mx-6 mt-6 rounded-lg border border-error/30 bg-error/10 p-3 text-[11px] uppercase tracking-widest text-error shadow-lg shadow-black/10">
              Serveur backend indisponible — démarre `uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000` ou `8001`, puis réessaie.
            </div>
          )}
          <header className="flex justify-between items-center px-6 py-3 w-full bg-[#1c1b1b] border-b border-[#4d4635]/15 backdrop-blur-2xl sticky top-0 z-50">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-primary animate-pulse shadow-[0_0_8px_#f2ca50]" />
                <span className="font-['Manrope'] uppercase tracking-widest text-[10px] text-[#f2ca50]">Status: Active</span>
              </div>
              <div className="h-4 w-px bg-[#4d4635]/40" />
              <span className="font-['Manrope'] uppercase tracking-widest text-[10px] text-[#e5e2e1]">NLP: Encrypted</span>
            </div>
            <div className="text-center">
              <h2 className="text-[#f2ca50] font-black tracking-widest text-xs lg:text-sm">GSAM | PRIVATE INTELLIGENCE</h2>
              <p className="text-[9px] text-on-surface-variant uppercase tracking-widest">Backend: {backendUrl}</p>
            </div>
            <div className="flex items-center gap-4">
              <button onClick={handleAbort} className="px-3 py-1.5 ghost-border text-[#ffb4ab] font-['Manrope'] uppercase tracking-widest text-[9px] hover:bg-error/10 transition-colors">ABORT</button>

            </div>
          </header>

          <div ref={chatCanvasRef} className="flex-1 overflow-y-auto px-6 py-8">
            <div className="max-w-5xl mx-auto space-y-8 pb-32">
              <div className="space-y-1">
                <h3 className="text-3xl lg:text-4xl font-bold tracking-tighter">Welcome back, <span className="text-primary">{userName}</span>.</h3>
                <p className="text-on-surface-variant text-sm tracking-wide opacity-80 uppercase tracking-widest">Protocol initiated. Agent awaiting vector instructions.</p>
              </div>

              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded bg-primary flex items-center justify-center shrink-0">
                  <span className="material-symbols-outlined text-on-primary">terminal</span>
                </div>
                <div className="space-y-4 flex-1">
                  <div>
                    <h4 className="text-[11px] font-bold text-primary uppercase tracking-widest">GSAM INTELLIGENCE</h4>
                    <p className="text-sm text-on-surface/90 mt-1">Protocol initiated. Awaiting vector instructions.</p>
                  </div>

                  <div id="step-tracker" className="space-y-3 pl-2 border-l border-outline-variant/20 ml-1">
                    {started
                      ? steps.map((step, index) => (
                          <div key={index} className="flex items-center gap-3">
                            {step.status === 'done' ? (
                              <>
                                <span className="material-symbols-outlined text-[#f2ca50] text-sm" style={{ fontVariationSettings: "'FILL' 1" }}>check_circle</span>
                                <p className="text-[11px] uppercase tracking-widest text-[#e5e2e1]/60">{step.name} {step.args ? `— ${step.args}` : ''}</p>
                              </>
                            ) : step.status === 'running' ? (
                              <>
                                <div className="w-2 h-2 rounded-full bg-[#f2ca50] animate-ping mx-1 flex-shrink-0" />
                                <p className="text-[11px] uppercase tracking-widest text-[#f2ca50] font-bold">{step.name}...</p>
                              </>
                            ) : (
                              <>
                                <div className="w-2 h-2 rounded-full bg-[#e5e2e1]/20 mx-1 flex-shrink-0" />
                                <p className="text-[11px] uppercase tracking-widest text-[#e5e2e1]/25">{step.name}</p>
                              </>
                            )}
                          </div>
                        ))
                      : null}
                  </div>

                  {resultData && (
                    <div className="bg-surface-container-high ghost-border p-6 rounded-xl shadow-2xl">
                      <div className="flex justify-between items-center mb-6">
                        <h4 className="text-[11px] font-bold text-primary uppercase tracking-widest">REAL-TIME TELEMETRY</h4>
                        <div className="flex gap-2">
                          <button onClick={exportCsv} className="text-[8px] bg-surface-container-lowest border border-outline-variant/30 px-2 py-1 rounded hover:border-primary/50 transition-all">EXPORT CSV</button>
                          <button onClick={exportPdf} className="text-[8px] bg-surface-container-lowest border border-outline-variant/30 px-2 py-1 rounded hover:border-primary/50 transition-all">EXPORT PDF</button>
                        </div>
                      </div>
                      <div className="grid grid-cols-3 gap-4">
                        <div className="p-3 bg-background/50 rounded ghost-border">
                          <p className="text-[9px] text-on-surface-variant uppercase tracking-widest mb-1">Endpoints</p>
                          <p className="text-xl font-bold text-primary">1,242</p>
                        </div>
                        <div className="p-3 bg-background/50 rounded ghost-border">
                          <p className="text-[9px] text-on-surface-variant uppercase tracking-widest mb-1">Vulnerabilities</p>
                          <p className="text-xl font-bold text-error">12</p>
                        </div>
                        <div className="p-3 bg-background/50 rounded ghost-border">
                          <p className="text-[9px] text-on-surface-variant uppercase tracking-widest mb-1">Uptime</p>
                          <p className="text-xl font-bold text-primary">99.8%</p>
                        </div>
                      </div>
                    </div>
                  )}
              {chatEvents.map((event, idx) => {
                if (event.type === 'user') {
                  return (
                    <div key={idx} className="flex justify-end">
                      <div className="bg-[#1c1b1b] border border-[#4d4635]/15 p-4 max-w-[80%] rounded-xl rounded-tr-none">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="bg-[#f2ca50]/20 text-[#f2ca50] text-[9px] px-2 py-0.5 rounded-full font-bold tracking-widest uppercase">CLIENT PRIORITY</span>
                          <span className="text-[9px] text-[#e5e2e1]/50 uppercase tracking-widest">{event.time}</span>
                        </div>
                        <p className="text-sm leading-relaxed text-[#e5e2e1]/90">{event.text}</p>
                      </div>
                    </div>
                  )
                }
                if (event.type === 'result') {
                  const entries = Object.entries(event.data || {})
                  return (
                    <div key={idx} className="flex items-start gap-4">
                      <div className="w-10 h-10 rounded bg-primary flex items-center justify-center shrink-0">
                        <span className="material-symbols-outlined text-on-primary">terminal</span>
                      </div>
                      <div className="space-y-3 flex-1">
                        <h4 className="text-[11px] font-bold text-primary uppercase tracking-widest">GSAM INTELLIGENCE — RESULT</h4>
                        <div className="bg-surface-container-high ghost-border p-5 rounded-xl">
                          <div className="grid grid-cols-2 gap-3">
                            {entries.map(([key, val]) => (
                              <div key={key} className="p-3 bg-background/50 rounded ghost-border">
                                <p className="text-[9px] text-on-surface-variant uppercase tracking-widest mb-1">{key.replace(/_/g, ' ')}</p>
                                <p className="text-sm font-bold text-primary break-all">{String(val)}</p>
                              </div>
                            ))}
                          </div>
                          <div className="flex gap-2 mt-4">
                            <button onClick={exportCsv} className="text-[8px] bg-surface-container-lowest border border-outline-variant/30 px-2 py-1 rounded hover:border-primary/50 transition-all">EXPORT CSV</button>
                            <button onClick={exportPdf} className="text-[8px] bg-surface-container-lowest border border-outline-variant/30 px-2 py-1 rounded hover:border-primary/50 transition-all">EXPORT PDF</button>
                          </div>
                        </div>
                      </div>
                    </div>
                  )
                }
                if (event.type === 'abort') {
                  return (
                    <div key={idx} className="text-center py-4">
                      <span className="text-[#ffb4ab] text-[11px] uppercase tracking-widest font-bold">
                        ⊗ MISSION ABORTED — Agent loop terminated by operator.
                      </span>
                    </div>
                  )
                }
                return (
                  <div key={idx} className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="col-span-1 md:col-span-2 bg-[#0e0e0e] border border-[#4d4635]/15 p-5 rounded-xl border-l-2 border-l-[#f2ca50]/30">
                      <h4 className="text-[10px] text-[#f2ca50] font-bold uppercase tracking-widest mb-4">NLP Analysis Card</h4>
                      <div className="flex flex-wrap gap-2">
                        <span className="px-3 py-1 bg-[#2a2a2a] text-[10px] text-[#e5e2e1]/70 border border-[#4d4635]/30 rounded uppercase tracking-widest">Intent: {event.intent}</span>
                        <span className="px-3 py-1 bg-[#2a2a2a] text-[10px] text-[#e5e2e1]/70 border border-[#4d4635]/30 rounded uppercase tracking-widest">Entity: {event.entity}</span>
                        <span className="px-3 py-1 bg-[#2a2a2a] text-[10px] text-[#f2ca50] border border-[#f2ca50]/20 rounded uppercase tracking-widest">Subtask: {event.subtask}</span>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </div>

          <div className="absolute bottom-0 left-0 w-full px-4 py-3 bg-gradient-to-t from-background via-background/95 to-transparent">
            <div className="max-w-5xl mx-auto space-y-2">
              <div className="flex items-center gap-3">
                <div className="flex-1 h-0.5 bg-surface-container-highest rounded-full overflow-hidden">
                  <div id="progress-fill" className="h-full gold-gradient" style={{ width: progressWidth, transition: 'width 0.6s ease' }} />
                </div>
                <span id="turn-label" className="text-[9px] font-bold text-primary tracking-widest uppercase shrink-0">TURN {currentTurn} / 15</span>
              </div>
              <div className="bg-surface-container-low/80 backdrop-blur-3xl ghost-border rounded-lg shadow-2xl flex items-center gap-2 px-3 py-2">
                <span className="material-symbols-outlined text-primary/50 text-[18px] shrink-0">terminal</span>
                <input
                  ref={commandInputRef}
                  id="command-input"
                  value={command}
                  onChange={(e) => setCommand(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault()
                      handleCommand()
                    }
                  }}
                  className="flex-1 bg-transparent border-none focus:ring-0 text-xs font-['Manrope'] tracking-wide placeholder:text-on-surface-variant/40 py-1"
                  placeholder="Awaiting mission parameters..."
                  type="text"
                />
                <label className="flex items-center gap-1.5 text-[9px] uppercase tracking-widest text-on-surface-variant shrink-0 cursor-pointer">
                  <input type="checkbox" checked={staleBrowser} onChange={(e) => setStaleBrowser(e.target.checked)} className="w-3 h-3 rounded border-outline-variant" />
                  Stale
                </label>
                <button id="command-btn" type="button" onClick={handleCommand} className="bg-gradient-to-r from-[#f2ca50] via-[#eac249] to-[#d4af37] text-[#3d2f00] font-black px-4 py-1.5 rounded-md flex items-center gap-1.5 hover:opacity-90 active:scale-95 transition-all shadow-lg shadow-[#f2ca50]/25 border border-[#f2ca50]/20 shrink-0">
                  <span className="font-['Manrope'] text-[9px] tracking-widest uppercase">CMD</span>
                  <span className="material-symbols-outlined text-xs">bolt</span>
                </button>
              </div>
            </div>
          </div>
        </main>

        <aside className="hidden xl:flex flex-col w-96 bg-surface-container-lowest border-l border-[#4d4635]/15 z-40 relative">
          <div className="p-4 border-b border-[#4d4635]/15 flex items-center justify-between bg-surface-container-low">
            <h3 className="text-[10px] font-bold text-primary tracking-widest uppercase">Live Browser Node</h3>
            <div className="flex gap-1">
              <div className="w-1.5 h-1.5 rounded-full bg-error/40" />
              <div className="w-1.5 h-1.5 rounded-full bg-primary/40" />
              <div className="w-1.5 h-1.5 rounded-full bg-on-surface/40" />
            </div>
          </div>
          <div className="p-4 bg-background/50 flex items-center gap-2">
            <div className="flex-1 bg-surface-container-highest px-3 py-1.5 rounded ghost-border flex items-center gap-2 truncate">
              <span className="material-symbols-outlined text-[14px] text-primary">lock</span>
              <span id="live-url" className="text-[10px] text-on-surface/70 truncate tracking-wide">{currentUrl || 'Awaiting connection...'}</span>
            </div>
            <button
              onClick={() => window.open(currentUrl || 'about:blank', '_blank')}
              title={currentUrl ? `Ouvrir ${currentUrl}` : 'Aucune URL active'}
              className={`p-1.5 rounded transition-all ${currentUrl ? 'text-primary hover:bg-primary/10 cursor-pointer' : 'text-on-surface-variant/30 cursor-not-allowed'}`}
            >
              <span className="material-symbols-outlined text-[18px]">open_in_new</span>
            </button>
          </div>
          <div className="flex-1 overflow-y-auto bg-surface-container-low/40 relative group">
            <div className="relative bg-black" style={{ minHeight: '320px' }}>
              {screenshot ? (
                <img id="live-screenshot" className="w-full h-auto block" alt="Capture navigateur" src={screenshot} />
              ) : (
                <div className="flex flex-col items-center justify-center h-64 text-on-surface-variant text-xs gap-2">
                  <span className="material-symbols-outlined text-primary/40 text-4xl animate-pulse">display_settings</span>
                  <span className="uppercase tracking-widest text-[10px]">En attente du navigateur...</span>
                </div>
              )}
              <div className="absolute inset-0 bg-gradient-to-b from-transparent via-primary/3 to-transparent h-20 w-full animate-scan pointer-events-none" />
            </div>
            <div className="px-4 pb-8 space-y-3">
              <h5 className="text-[9px] font-bold text-on-surface-variant uppercase tracking-widest">Sequence Log</h5>
              <div id="sequence-log" className="space-y-2">
                {logs.map((log, index) => (
                  <div
                    key={index}
                    className={`text-[10px] p-2 bg-surface-container-lowest rounded border-l-2 font-mono ${
                      log.faded
                        ? 'border-on-surface/20 opacity-50'
                        : log.active
                        ? 'border-[#f2ca50] bg-[#f2ca50]/5'
                        : 'border-primary/40'
                    }`}
                  >
                    <span className="text-primary/60">[{log.time}]</span> {log.text}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </aside>
      </div>

      {toastVisible && (
        <div className="fixed top-8 right-8 z-[100] transition-transform duration-500 ease-out flex items-center gap-4 bg-surface-container-high border-l-4 border-primary p-5 rounded shadow-2xl max-w-sm">
          <div className="p-2 rounded-full bg-primary/10 text-primary">
            <span className="material-symbols-outlined">task_alt</span>
          </div>
          <div>
            <p className="text-xs font-bold uppercase tracking-widest">MISSION COMPLETE</p>
            <p className="text-[10px] text-on-surface-variant mt-1">Protocol GSAM-9 has finished archival.</p>
          </div>
        </div>
      )}

      {showModal && (
        <div id="safety-modal" className="fixed inset-0 z-[200] bg-black/70 backdrop-blur-sm flex items-center justify-center">
          <div className="bg-[#2a2a2a] max-w-md w-full mx-4 rounded-xl p-8 border border-[#4d4635]/15 shadow-2xl shadow-black">
            <div className="text-[#f2ca50] text-5xl text-center mb-4 material-symbols-outlined">gpp_maybe</div>
            <h3 className="text-[#f2ca50] text-[11px] font-bold uppercase tracking-widest text-center mb-3">ACTION REQUIRES CONFIRMATION</h3>
            <p id="safety-explanation" className="text-[#e5e2e1]/70 text-sm text-center leading-relaxed mb-8">
              {safetyMessage || 'The agent is about to perform a sensitive action. This cannot be undone. Confirm to proceed.'}
            </p>
            <div className="flex gap-3 justify-center">
              <button id="safety-abort-btn" onClick={handleAbort} className="text-[#ffb4ab] border border-[#ffb4ab]/20 px-6 py-2.5 rounded-lg text-[10px] uppercase tracking-widest font-bold hover:bg-[#ffb4ab]/10 transition-colors">
                ABORT
              </button>
              <button id="safety-confirm-btn" onClick={handleConfirm} className="bg-gradient-to-r from-[#f2ca50] via-[#eac249] to-[#d4af37] text-[#3d2f00] px-6 py-2.5 rounded-lg text-[10px] uppercase tracking-widest font-bold hover:opacity-90 transition-opacity shadow-2xl shadow-[#f2ca50]/35 border border-[#f2ca50]/20">
                CONFIRM
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App