import { useEffect, useMemo, useRef, useState } from 'react'
import { Users, Clock, MoreVertical } from 'lucide-react'
// Lazily import socket.io-client inside effects to avoid type resolution issues during build/typecheck

function useDemoTimer(initialSeconds = 300) {
  const [seconds, setSeconds] = useState(initialSeconds)
  const [state, setState] = useState<'running' | 'paused' | 'expired'>('running')

  useEffect(() => {
    if (state !== 'running') return
    const id = setInterval(() => {
      setSeconds((s) => {
        if (s <= 1) {
          setState('expired')
          return 0
        }
        return s - 1
      })
    }, 1000)
    return () => clearInterval(id)
  }, [state])

  const minutes = Math.floor(seconds / 60)
  const rem = seconds % 60

  const reset = (s = initialSeconds) => {
    setSeconds(s)
    setState('running')
  }

  const setTime = (s: number) => {
    setSeconds(s)
  }

  return { minutes, rem, state, setState, reset, setTime, setSeconds }
}

type Config = {
  gameMode: string
  roundTimer: number
  maxPlayers: number
  inkLimit: 'Low' | 'Medium' | 'High' | 'Unlimited'
  teamsEnabled: boolean
  visibility: 'Public' | 'Private'
  password: string
  contentSafety: 'SFW' | 'NSFW'
  customPrompt: string
}

// Player list will be wired later from real client addresses
type Player = { id: string; name: string; status: 'online' | 'idle' }

export default function AdminPanel() {
  const { minutes, rem, state, setState, reset, setSeconds } = useDemoTimer(300)
  const [duration, setDuration] = useState<number>(300)
  const [sessionId, setSessionId] = useState<string>('')
  const socketRef = useRef<any>(null)

  const [config, setConfig] = useState<Config>({
    gameMode: 'Speed Sketch',
    roundTimer: 300,
    maxPlayers: 8,
    inkLimit: 'Medium',
    teamsEnabled: false,
    visibility: 'Public',
    password: '',
    contentSafety: 'SFW',
    customPrompt: '',
  })

  const [sessionLocked, setSessionLocked] = useState(false)

  const [customTimer, setCustomTimer] = useState<number>(90)
  const [timerPreset, setTimerPreset] = useState<string>('300')

  // Compute a flat, human-readable config for persistence
  const adminKv = useMemo(() => ({
    Timer: config.roundTimer,
    TimerPreset: String(timerPreset),
    'Game Mode': config.gameMode,
    'Max Players': config.maxPlayers,
    'Ink Limit': config.inkLimit,
    Teams: config.teamsEnabled ? 'enabled' : 'disabled',
    Visibility: config.visibility,
    Password: config.password,
    'Custom Prompt': config.customPrompt,
    'Content Mode': config.contentSafety,
    Session: sessionLocked ? 'Locked' : 'Open',
  }), [config, timerPreset, sessionLocked])

  // Load initial values from server state (preferred), fallback to AdminConfig.json
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        // Try full state first
        let dataState: any | null = null
        try {
          const r = await fetch('/api/state')
          if (r.ok) dataState = await r.json()
        } catch {}

        const data = dataState?.config ? dataState.config : await (async () => {
          const res = await fetch('/api/admin-config')
          if (!res.ok) return null
          return await res.json()
        })()
        if (!data) return
        if (cancelled) return
        if (dataState?.sessionId) setSessionId(String(dataState.sessionId))
        if (typeof dataState?.players?.count === 'number') setConnectedCount(Number(dataState.players.count) || 0)
        // Map into local state
        setTimerPreset(String(data['TimerPreset'] ?? '300'))
        const timer = Number(data['Timer'] ?? dataState?.timer?.duration ?? 300)
        setConfig((c) => ({
          ...c,
          roundTimer: Number.isFinite(timer) ? timer : 300,
          gameMode: typeof data['Game Mode'] === 'string' ? data['Game Mode'] : c.gameMode,
          maxPlayers: Number.isFinite(Number(data['Max Players'])) ? Number(data['Max Players']) : c.maxPlayers,
          inkLimit: (data['Ink Limit'] as any) ?? c.inkLimit,
          teamsEnabled: String(data['Teams']).toLowerCase() === 'enabled',
          visibility: (data['Visibility'] as any) ?? c.visibility,
          password: typeof data['Password'] === 'string' ? data['Password'] : c.password,
          customPrompt: typeof data['Custom Prompt'] === 'string' ? data['Custom Prompt'] : c.customPrompt,
          contentSafety: (data['Content Mode'] as any) ?? c.contentSafety,
        }))
        setSessionLocked(String(data['Session']).toLowerCase() === 'locked')
      } catch (_) {
        // No-op: server route may not exist yet
      }
    })()
    return () => { cancelled = true }
  }, [])

  // Debounced persist to server
  const postTimer = useRef<number | null>(null)
  const lastSent = useRef<string>('')
  useEffect(() => {
    const body = JSON.stringify(adminKv)
    if (lastSent.current === body) return
    if (postTimer.current) window.clearTimeout(postTimer.current)
    postTimer.current = window.setTimeout(async () => {
      try {
        const res = await fetch('/api/admin-config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body,
        })
        if (res.ok) {
          lastSent.current = body
        }
      } catch (_) {
        // Ignore network errors (e.g., offline or route missing)
      }
    }, 800)
    return () => {
      if (postTimer.current) window.clearTimeout(postTimer.current)
    }
  }, [adminKv])

  useEffect(() => {
    setDuration(config.roundTimer)
    reset(config.roundTimer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.roundTimer])

  const [connectedCount, setConnectedCount] = useState<number>(0)
  const [players, setPlayers] = useState<Player[]>([])
  const [openMenu, setOpenMenu] = useState<string | null>(null)

  const refreshPlayers = async () => {
    try {
      const res = await fetch('/api/players')
      if (!res.ok) return
      const data = await res.json()
      const list: Array<{ id: string; address?: string }> = data?.list || []
      setConnectedCount(Number(data?.count) || list.length || 0)
      setPlayers(
        list.map((p) => ({
          id: p.id,
          name: p.address || p.id,
          status: 'online',
        }))
      )
    } catch (_) {
      // ignore
    }
  }

  const kickPlayer = async (id: string) => {
    try {
      const res = await fetch('/api/players/kick', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id })
      })
      if (res.ok) {
        await refreshPlayers()
      }
    } catch (_) {
      // ignore
    }
  }

  // Live updates via socket.io (players, config, timer, state)
  useEffect(() => {
    let socket: any = null
    let scriptEl: HTMLScriptElement | null = null

    const onReady = () => {
      const io = (window as any).io
      if (!io) return
      socket = io({ transports: ['websocket'] })
      socketRef.current = socket

      socket.on('connect', () => {
        console.log('Admin panel connected to socket:', socket.id)
      })

      socket.on('state:init', async (payload: any) => {
        if (payload?.sessionId) setSessionId(String(payload.sessionId))
        if (payload?.players?.count !== undefined) setConnectedCount(Number(payload.players.count) || 0)
        // initial players list
        await refreshPlayers()
        if (payload?.config) {
          const data = payload.config
          setTimerPreset(String(data['TimerPreset'] ?? '300'))
          const timer = Number(data['Timer'] ?? payload?.timer?.duration ?? 300)
          setConfig((c) => ({
            ...c,
            roundTimer: Number.isFinite(timer) ? timer : 300,
            gameMode: typeof data['Game Mode'] === 'string' ? data['Game Mode'] : c.gameMode,
            maxPlayers: Number.isFinite(Number(data['Max Players'])) ? Number(data['Max Players']) : c.maxPlayers,
            inkLimit: (data['Ink Limit'] as any) ?? c.inkLimit,
            teamsEnabled: String(data['Teams']).toLowerCase() === 'enabled',
            visibility: (data['Visibility'] as any) ?? c.visibility,
            password: typeof data['Password'] === 'string' ? data['Password'] : c.password,
            customPrompt: typeof data['Custom Prompt'] === 'string' ? data['Custom Prompt'] : c.customPrompt,
            contentSafety: (data['Content Mode'] as any) ?? c.contentSafety,
          }))
          setSessionLocked(String(data['Session']).toLowerCase() === 'locked')
        }
      })

      const applyConfig = (data: any) => {
        if (!data) return
        setTimerPreset(String(data['TimerPreset'] ?? '300'))
        const timer = Number(data['Timer'] ?? 300)
        setConfig((c) => ({
          ...c,
          roundTimer: Number.isFinite(timer) ? timer : c.roundTimer,
          gameMode: typeof data['Game Mode'] === 'string' ? data['Game Mode'] : c.gameMode,
          maxPlayers: Number.isFinite(Number(data['Max Players'])) ? Number(data['Max Players']) : c.maxPlayers,
          inkLimit: (data['Ink Limit'] as any) ?? c.inkLimit,
          teamsEnabled: String(data['Teams']).toLowerCase() === 'enabled',
          visibility: (data['Visibility'] as any) ?? c.visibility,
          password: typeof data['Password'] === 'string' ? data['Password'] : c.password,
          customPrompt: typeof data['Custom Prompt'] === 'string' ? data['Custom Prompt'] : c.customPrompt,
          contentSafety: (data['Content Mode'] as any) ?? c.contentSafety,
        }))
        setSessionLocked(String(data['Session']).toLowerCase() === 'locked')
      }

      socket.on('config:update', applyConfig)
      socket.on('admin-config:update', applyConfig)
      socket.on('players:update', async (p: any) => {
        setConnectedCount(Number(p?.count) || 0)
        await refreshPlayers()
      })
      socket.on('timer:update', (snapshot: any) => {
        if (!snapshot) return
        const { state: timerState, remaining, duration: dur } = snapshot
        // Sync admin timer display with server
        if (Number.isFinite(remaining)) {
          setSeconds(remaining)
          if (Number.isFinite(dur) && dur > 0) setDuration(dur)
        }
        if (timerState === 'running') setState('running')
        else if (timerState === 'paused') setState('paused')
        else if (timerState === 'expired') setState('expired')
      })
    }

    // Load the socket.io client from the server to avoid module/type issues
    // It is served automatically by the socket.io server under this path
    scriptEl = document.createElement('script')
    scriptEl.src = '/socket.io/socket.io.js'
    scriptEl.async = true
    scriptEl.onload = onReady
    document.body.appendChild(scriptEl)

    return () => {
      if (socket) {
        try { socket.close() } catch {}
      }
      if (scriptEl) {
        try { document.body.removeChild(scriptEl) } catch {}
      }
    }
  }, [])

  return (
    <div className="dashboard-container">
      <div className="doodle-header">
        <h1>Doodle Party</h1>
      </div>

      <div className="glass-panel p-6 uniform-panel" style={{ gridColumn: '1 / -1' }}>
        <h3 className="panel-title" style={{ marginBottom: 8 }}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
            <Users style={{ width: 24, height: 24 }} /> Player list
          </span>
          {sessionLocked && (
            <span className="badge" style={{ marginLeft: 10 }}>
              Locked
            </span>
          )}
        </h3>

        <div className="form-group-inline" style={{ marginBottom: 12, justifyContent: 'space-between' }}>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label>Connected</label>
            <input type="text" value={String(connectedCount)} readOnly />
          </div>
          <button className="btn btn-secondary" onClick={() => setSessionLocked(v => !v)}>
            {sessionLocked ? 'Unlock' : 'Lock'}
          </button>
        </div>

        <div className="scroll-area" style={{ maxHeight: 180, overflow: 'auto' }}>
          {players.length === 0 ? (
            <div className="muted" style={{ padding: 8 }}>
              Player names will appear here once real clients are wired up.
            </div>
          ) : (
            players.map((p) => (
              <div
                key={p.id}
                className="flex items-center justify-between border rounded p-3 mb-2"
                style={{ borderColor: 'rgba(255,255,255,0.6)' }}
              >
                <div className="flex items-center gap-3">
                  <span className={`status-dot ${p.status}`} />
                  <div>
                    <div className="flex items-center gap-2" style={{ fontWeight: 400 }}>
                      {p.name}
                    </div>
                    <small className="muted">{p.status === 'online' ? 'Online' : 'Idle'}</small>
                  </div>
                </div>

                <div style={{ position: 'relative' }}>
                  <button
                    className="btn btn-secondary btn-icon"
                    onClick={() => setOpenMenu(openMenu === p.id ? null : p.id)}
                  >
                    <MoreVertical />
                  </button>

                  {openMenu === p.id && (
                    <div className="menu glass-panel p-4" style={{ position: 'absolute', right: 0, top: '110%', minWidth: 200 }}>
                      <button
                        className="btn btn-block"
                        onClick={() => {
                          kickPlayer(p.id)
                          setOpenMenu(null)
                        }}
                      >
                        Kick
                      </button>
                      <button className="btn btn-block" onClick={() => setOpenMenu(null)}>
                        Close
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="panel-grid">
        <div className="glass-panel p-6 uniform-panel">
          <h2 className="panel-title">
            <Clock style={{ width: 28, height: 28 }} /> Time:
          </h2>

          <div className="form-group">
            <label>time setter + time</label>
            <div className="form-group-inline">
              <select
                value={timerPreset}
                onChange={(e) => {
                  const val = e.target.value
                  setTimerPreset(val)
                  if (val === 'custom') {
                    setConfig((c) => ({ ...c, roundTimer: customTimer }))
                    setDuration(customTimer)
                    reset(customTimer)
                  } else {
                    const d = Number(val)
                    setConfig((c) => ({ ...c, roundTimer: d }))
                    setDuration(d)
                    reset(d)
                  }
                }}
              >
                <option value="30">0:30</option>
                <option value="60">1:00</option>
                <option value="120">2:00</option>
                <option value="300">5:00</option>
                <option value="600">10:00</option>
                <option value="custom">Custom</option>
              </select>

              {timerPreset === 'custom' && (
                <input
                  type="number"
                  min={10}
                  step={5}
                  value={customTimer}
                  onChange={(e) => {
                    const v = Math.max(10, Number(e.target.value) || 10)
                    setCustomTimer(v)
                    setConfig((c) => ({ ...c, roundTimer: v }))
                    setDuration(v)
                    reset(v)
                  }}
                />
              )}
            </div>
          </div>

          <div className="timer-value">
            {minutes}:{String(rem).padStart(2, '0')}
          </div>

          <div className="form-group-inline">
            <button
              className="btn btn-primary"
              onClick={async () => {
                setState('running')
                try { await fetch('/api/timer', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: 'start' }) }) } catch {}
              }}
            >
              start
            </button>
            <button
              className="btn btn-secondary"
              onClick={async () => {
                setState('paused')
                try { await fetch('/api/timer', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: 'pause' }) }) } catch {}
              }}
            >
              pause
            </button>
            <button
              className="btn btn-secondary"
              onClick={async () => {
                setState('running')
                reset(duration)
                try { 
                  await fetch('/api/timer', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: 'reset', seconds: duration }) })
                  await fetch('/api/timer', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: 'start' }) })
                } catch {}
              }}
            >
              reset & start
            </button>
            <button
              className="btn btn-danger"
              onClick={async () => {
                setState('expired')
                try { await fetch('/api/timer', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: 'stop' }) }) } catch {}
              }}
            >
              stop
            </button>
          </div>
        </div>

        <div className="glass-panel p-4 uniform-panel">
          <h2 className="panel-title">Canvas Actions</h2>
          <button
            className="btn btn-danger btn-block"
            onClick={() => {
              if (socketRef.current && socketRef.current.connected) {
                socketRef.current.emit('quickdraw.clear')
                console.log('Admin: Emitted quickdraw.clear event')
              } else {
                console.warn('Admin: Socket not connected, cannot clear canvas')
              }
            }}
          >
            Clear Canvas
          </button>
        </div>

        <div className="glass-panel p-4 uniform-panel">
          <h2 className="panel-title">Game Mode:</h2>
          <div className="mode-chip-group">
            {['Classic', 'Speed', 'Story'].map((label) => (
              <button
                key={label}
                className={`btn mode-btn ${
                  (label === 'Classic' && config.gameMode === 'Classic') ||
                  (label === 'Speed' && (config.gameMode === 'Speed' || config.gameMode === 'Speed Sketch')) ||
                  (label === 'Story' && (config.gameMode === 'Story' || config.gameMode === 'Story Canvas'))
                    ? 'active'
                    : ''
                }`}
                onClick={() => setConfig((c) => ({ ...c, gameMode: label === 'Speed' ? 'Speed' : label }))}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="glass-panel p-6 uniform-panel">
          <h2 className="panel-title">Game Configuration</h2>

          <div className="form-group">
            <label>set max players</label>
            <select
              value={String(config.maxPlayers)}
              onChange={(e) => setConfig((c) => ({ ...c, maxPlayers: Number(e.target.value) }))}
            >
              <option value={4}>4</option>
              <option value={8}>8</option>
              <option value={16}>16</option>
              <option value={32}>32</option>
              <option value={-1}>unlimited</option>
            </select>
          </div>

          <div className="form-group">
            <label>set ink limit (Low, Medium, High, Unlimited)</label>
            <select value={config.inkLimit} onChange={(e) => setConfig((c) => ({ ...c, inkLimit: e.target.value as any }))}>
              <option>Low</option>
              <option>Medium</option>
              <option>High</option>
              <option>Unlimited</option>
            </select>
          </div>

          <div className="form-group-inline">
            <div className="form-group">
              <label>Enable/ disable teams</label>
              <select value={String(config.teamsEnabled)} onChange={(e) => setConfig((c) => ({ ...c, teamsEnabled: e.target.value === 'true' }))}>
                <option value={'false'}>disabled</option>
                <option value={'true'}>enabled</option>
              </select>
            </div>

            <div className="form-group">
              <label>Private vs Public</label>
              <select value={config.visibility} onChange={(e) => setConfig((c) => ({ ...c, visibility: e.target.value as any }))}>
                <option>Public</option>
                <option>Private</option>
              </select>
            </div>
          </div>

          {config.visibility === 'Private' && (
            <div className="form-group">
              <label>Password-protected</label>
              <input
                type="password"
                value={config.password}
                onChange={(e) => setConfig((c) => ({ ...c, password: e.target.value }))}
                placeholder="enter password"
              />
            </div>
          )}
        </div>

        <div className="glass-panel p-6 uniform-panel">
          <h2 className="panel-title">Custom prompt for drawing</h2>
          <textarea
            placeholder="write here"
            value={config.customPrompt}
            onChange={(e) => setConfig((c) => ({ ...c, customPrompt: e.target.value }))}
          />
        </div>

        <div className="glass-panel p-6 uniform-panel">
          <h2 className="panel-title">Moderation Settings</h2>
          <div className="form-group">
            <label>Content Mode</label>
            <div className="mode-chip-group">
              <button
                className={`btn mode-btn ${config.contentSafety === 'SFW' ? 'active' : ''}`}
                onClick={() => setConfig((c) => ({ ...c, contentSafety: 'SFW' }))}
              >
                SFW
              </button>
              <button
                className={`btn mode-btn ${config.contentSafety === 'NSFW' ? 'active' : ''}`}
                onClick={() => setConfig((c) => ({ ...c, contentSafety: 'NSFW' }))}
              >
                NSFW
              </button>
            </div>
          </div>
        </div>

        <div className="glass-panel p-6 uniform-panel">
          <h2 className="panel-title">Session Info</h2>

          <div className="form-group-inline">
            <div className="form-group">
              <label>Mode</label>
              <input type="text" value={config.gameMode} readOnly />
            </div>
            <div className="form-group">
              <label>Max Players</label>
              <input type="text" value={config.maxPlayers === -1 ? 'unlimited' : String(config.maxPlayers)} readOnly />
            </div>
          </div>

          <div className="form-group-inline">
            <div className="form-group">
              <label>Visibility</label>
              <input type="text" value={config.visibility} readOnly />
            </div>
            <div className="form-group">
              <label>Session</label>
              <input type="text" value={sessionLocked ? 'Locked' : 'Open'} readOnly />
            </div>
          </div>

          <div className="form-group-inline">
            <div className="form-group">
              <label>Content Mode</label>
              <input type="text" value={config.contentSafety} readOnly />
            </div>
            <div className="form-group">
              <label>Ink Limit</label>
              <input type="text" value={config.inkLimit} readOnly />
            </div>
          </div>

          <div className="form-group-inline">
            <div className="form-group">
              <label>Teams</label>
              <input type="text" value={config.teamsEnabled ? 'enabled' : 'disabled'} readOnly />
            </div>
            <div className="form-group">
              <label>Round length</label>
              <input type="text" value={`${Math.floor(config.roundTimer/60)}:${String(config.roundTimer%60).padStart(2,'0')}`} readOnly />
            </div>
          </div>

          <div className="form-group">
            <label>Connected</label>
            <input type="text" value={String(connectedCount)} readOnly />
          </div>

          <div className="form-group">
            <label>Session ID</label>
            <input type="text" value={sessionId || '—'} readOnly />
          </div>
        </div>
      </div>
    </div>
  )
}
