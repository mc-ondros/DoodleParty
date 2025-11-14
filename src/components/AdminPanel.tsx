import { useEffect, useMemo, useState } from 'react'
import { Users, PenLine, Trash2, TrendingUp, Clock, Palette, MoreVertical, Crown, Lock } from 'lucide-react'

type Stat = { label: string; value: string | number; icon: React.ComponentType<{ className?: string }>; description?: string }

function useTheme() {
  const [theme, setThemeState] = useState<string>(() => localStorage.getItem('preferredTheme') || 'modern')
  useEffect(() => {
    const root = document.documentElement
    root.classList.remove('theme-modern', 'theme-eink', 'theme-neon', 'theme-pastel', 'theme-highcontrast')
    root.classList.add(`theme-${theme}`)
    localStorage.setItem('preferredTheme', theme)
  }, [theme])
  return { theme, setTheme: setThemeState }
}

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
  return { minutes, rem, state, setState, reset: (s = initialSeconds) => { setSeconds(s); setState('running') } }
}

export default function AdminPanel() {
  const { theme, setTheme } = useTheme()
  const { minutes, rem, state, setState, reset } = useDemoTimer(300)
  const [duration, setDuration] = useState(300)

  // configuration + moderation + prompts (demo state)
  const [config, setConfig] = useState({
    gameMode: 'Speed Sketch',
    roundTimer: 300, // seconds
    maxPlayers: 8 as number | -1,
    inkLimit: 'Medium' as 'Low' | 'Medium' | 'High' | 'Unlimited',
    teamsEnabled: false,
    visibility: 'Public' as 'Public' | 'Private',
    password: '',
    aiStrictness: 'Normal' as 'Relaxed' | 'Normal' | 'Strict',
    autoKick: 5 as number | -1, // -1 disabled
    difficulty: 'Medium' as 'Easy' | 'Medium' | 'Hard',
    category: 'Random',
    customPrompt: '',
  })

  const [sessionLocked, setSessionLocked] = useState(false)

  // keep session duration in sync with configuration round timer
  useEffect(() => {
    setDuration(config.roundTimer)
    reset(config.roundTimer)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.roundTimer])

  const stats: Stat[] = useMemo(() => ([
    { label: 'ACTIVE USERS', value: 24, icon: Users },
    { label: 'TOTAL STROKES', value: 1243, icon: PenLine },
    { label: 'REMOVED', value: 47, icon: Trash2 },
    { label: 'AVG CONFIDENCE', value: '87%', icon: TrendingUp },
    { label: 'SESSIONS', value: 8, icon: Clock },
  ]), [])

  const themes = [
    { name: 'modern', label: 'Modern' },
    { name: 'eink', label: 'E-Ink' },
    { name: 'neon', label: 'Neon' },
    { name: 'pastel', label: 'Pastel' },
    { name: 'highcontrast', label: 'High Contrast' },
  ]

  // demo players
  type Player = { id: string; name: string; isHost?: boolean; status: 'online' | 'idle' }
  const [players, setPlayers] = useState<Player[]>([
    { id: '1', name: 'Alex', isHost: true, status: 'online' },
    { id: '2', name: 'Jamie', status: 'idle' },
    { id: '3', name: 'Riley', status: 'online' },
  ])
  const [newPlayer, setNewPlayer] = useState('')
  const [openMenu, setOpenMenu] = useState<string | null>(null)

  const addPlayer = () => {
    const name = newPlayer.trim()
    if (!name) return
    setPlayers((ps) => [...ps, { id: Math.random().toString(36).slice(2), name, status: 'online' }])
    setNewPlayer('')
  }
  const kickPlayer = (id: string) => setPlayers((ps) => ps.filter((p) => p.id !== id))
  const transferHost = (id: string) => setPlayers((ps) => ps.map((p) => ({ ...p, isHost: p.id === id })))

  return (
    <div className="dashboard-container">
      <div className="glass-panel p-6 dashboard-header">
        <h1>Admin Dashboard</h1>
        <p>Real-time monitoring and controls for drawing sessions.</p>
      </div>

      <div className="stats-grid">
        {stats.map(({ label, value, icon: Icon }) => (
          <div className="glass-panel p-6 stat-card" key={label}>
            <div className="stat-card-content">
              <p className="stat-card-label">{label}</p>
              <p className="stat-card-value">{value}</p>
            </div>
            <Icon className="stat-card-icon" />
          </div>
        ))}
      </div>

      {/* Second row mirrors first: 3 glass cards */}
      <div className="stats-grid">
        {/* Controls */}
        <div className="glass-panel p-6 control-panel">
          <h3 className="flex items-center gap-2"><Clock className="stat-card-icon" /> Session Controls</h3>
          <label className="form-group">
            <span className="stat-card-label">Round Timer</span>
            <select
              value={config.roundTimer}
              onChange={(e) => setConfig((c) => ({ ...c, roundTimer: Number(e.target.value) }))}
            >
              <option value={30}>0:30</option>
              <option value={60}>1:00</option>
              <option value={120}>2:00</option>
              <option value={300}>5:00</option>
              <option value={600}>10:00</option>
            </select>
          </label>
          <div className="form-group-inline">
            <button className="btn btn-primary" onClick={() => setState('running')}>Start</button>
            <button className="btn btn-secondary" onClick={() => setState('paused')}>Pause</button>
            <button className="btn btn-secondary" onClick={() => reset(duration)}>Reset</button>
            <button className="btn btn-danger" onClick={() => setState('expired')}>Stop</button>
          </div>
          <div className="form-group-inline">
            <button
              className={`btn ${sessionLocked ? 'btn-warning' : 'btn-secondary'}`}
              onClick={() => setSessionLocked((v) => !v)}
              title={sessionLocked ? 'Unlock to allow new players' : 'Lock to prevent new joins'}
            >
              <Lock className="stat-card-icon" /> {sessionLocked ? 'Unlock Session' : 'Lock Session'}
            </button>
          </div>
        </div>

        {/* Timer */}
        <div className="glass-panel timer-display">
          <p className="timer-label">Time Remaining</p>
          <div className="timer-value">{minutes}:{String(rem).padStart(2, '0')}</div>
          <p className={`timer-status ${state}`}>{state === 'running' ? 'Running' : state === 'paused' ? 'Paused' : 'Expired'}</p>
        </div>

        {/* Prompt & Theme */}
        <div className="glass-panel p-6">
          <h3 className="flex items-center gap-2"><Palette className="stat-card-icon" /> Prompt & Theme</h3>
          <div className="theme-selector" style={{ marginTop: 0 }}>
            {themes.map(t => (
              <button
                key={t.name}
                className={`btn theme-option ${theme === t.name ? 'active' : ''}`}
                onClick={() => setTheme(t.name)}
              >{t.label}</button>
            ))}
          </div>
          <div className="form-group" style={{ marginTop: 'var(--space-md)' }}>
            <label className="stat-card-label">Difficulty</label>
            <select value={config.difficulty} onChange={(e) => setConfig(c => ({ ...c, difficulty: e.target.value as any }))}>
              <option>Easy</option>
              <option>Medium</option>
              <option>Hard</option>
            </select>
          </div>
          <div className="form-group">
            <label className="stat-card-label">Theme / Category</label>
            <select value={config.category} onChange={(e) => setConfig(c => ({ ...c, category: e.target.value }))}>
              <option>Animals</option>
              <option>Food</option>
              <option>Sports</option>
              <option>Random</option>
            </select>
          </div>
          <div className="form-group">
            <label className="stat-card-label">Custom Prompt</label>
            <textarea
              placeholder="Override AI-generated prompt..."
              value={config.customPrompt}
              onChange={(e) => setConfig(c => ({ ...c, customPrompt: e.target.value }))}
            />
          </div>
        </div>
      </div>

      {/* Game Configuration & Moderation */}
      <div className="stats-grid">
        {/* Game Configuration */}
        <div className="glass-panel p-6">
          <h3>Game Configuration</h3>
          <div className="form-group">
            <label className="stat-card-label">Game Mode</label>
            <select value={config.gameMode} onChange={(e) => setConfig(c => ({ ...c, gameMode: e.target.value }))}>
              <option>Speed Sketch</option>
              <option>Battle Royale</option>
              <option>Story Canvas</option>
            </select>
          </div>
          <div className="form-group-inline">
            <div className="form-group">
              <label className="stat-card-label">Max Players</label>
              <select value={String(config.maxPlayers)} onChange={(e) => setConfig(c => ({ ...c, maxPlayers: Number(e.target.value) }))}>
                <option value={4}>4</option>
                <option value={8}>8</option>
                <option value={16}>16</option>
                <option value={32}>32</option>
                <option value={-1}>Unlimited</option>
              </select>
            </div>
            <div className="form-group">
              <label className="stat-card-label">Ink Limit</label>
              <select value={config.inkLimit} onChange={(e) => setConfig(c => ({ ...c, inkLimit: e.target.value as any }))}>
                <option>Low</option>
                <option>Medium</option>
                <option>High</option>
                <option>Unlimited</option>
              </select>
            </div>
          </div>
          <div className="form-group-inline">
            <div className="form-group">
              <label className="stat-card-label">Teams</label>
              <select value={String(config.teamsEnabled)} onChange={(e) => setConfig(c => ({ ...c, teamsEnabled: e.target.value === 'true' }))}>
                <option value={'false'}>Disabled</option>
                <option value={'true'}>Enabled</option>
              </select>
            </div>
            <div className="form-group">
              <label className="stat-card-label">Visibility</label>
              <select value={config.visibility} onChange={(e) => setConfig(c => ({ ...c, visibility: e.target.value as any }))}>
                <option>Public</option>
                <option>Private</option>
              </select>
            </div>
          </div>
          {config.visibility === 'Private' && (
            <div className="form-group">
              <label className="stat-card-label">Session Password</label>
              <input type="password" value={config.password} onChange={(e) => setConfig(c => ({ ...c, password: e.target.value }))} placeholder="Required to join" />
            </div>
          )}
        </div>

        {/* Moderation Settings */}
        <div className="glass-panel p-6">
          <h3>Moderation Settings</h3>
          <div className="form-group">
            <label className="stat-card-label">AI Strictness</label>
            <select value={config.aiStrictness} onChange={(e) => setConfig(c => ({ ...c, aiStrictness: e.target.value as any }))}>
              <option>Relaxed</option>
              <option>Normal</option>
              <option>Strict</option>
            </select>
          </div>
          <div className="form-group">
            <label className="stat-card-label">Auto-Kick After Violations</label>
            <select value={String(config.autoKick)} onChange={(e) => setConfig(c => ({ ...c, autoKick: Number(e.target.value) }))}>
              <option value={-1}>Disabled</option>
              <option value={3}>3</option>
              <option value={5}>5</option>
              <option value={10}>10</option>
            </select>
          </div>
        </div>

        {/* Pre-Game Setup (Session Creation) - simple status */}
        <div className="glass-panel p-6">
          <h3>Session Setup</h3>
          <p className="muted">Configure options, then use Session Controls to start the round.</p>
          <div className="form-group-inline">
            <div className="form-group">
              <label className="stat-card-label">Current Mode</label>
              <input type="text" value={config.gameMode} readOnly />
            </div>
            <div className="form-group">
              <label className="stat-card-label">Players Limit</label>
              <input type="text" value={config.maxPlayers === -1 ? 'Unlimited' : String(config.maxPlayers)} readOnly />
            </div>
          </div>
          <div className="form-group-inline">
            <div className="form-group">
              <label className="stat-card-label">Session</label>
              <input type="text" value={sessionLocked ? 'Locked' : 'Open'} readOnly />
            </div>
            <div className="form-group">
              <label className="stat-card-label">Visibility</label>
              <input type="text" value={config.visibility} readOnly />
            </div>
          </div>
        </div>
      </div>

      {/* Players list - full-width card below */}
      <div className="glass-panel p-6" style={{ gridColumn: '1 / -1' }}>
        <div className="flex items-center justify-between mb-3">
          <h3 className="flex items-center gap-2"><Users className="stat-card-icon" /> Players {sessionLocked && <span className="badge" title="Session is locked">Locked</span>}</h3>
          <div className="form-group-inline">
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="stat-card-label">Add Player</label>
              <input
                type="text"
                value={newPlayer}
                onChange={(e) => setNewPlayer(e.target.value)}
                placeholder="Player name"
                disabled={sessionLocked}
              />
            </div>
            <button className="btn btn-primary" onClick={addPlayer} disabled={sessionLocked}>Add</button>
          </div>
        </div>

        <div className="scroll-area" style={{ maxHeight: 320, overflow: 'auto' }}>
          {players.map(p => (
            <div key={p.id} className="flex items-center justify-between border rounded p-3 mb-2">
              <div className="flex items-center gap-3">
                <span className={`status-dot ${p.status}`} aria-label={p.status} title={p.status}></span>
                <div>
                  <div className="font-semibold flex items-center gap-2">
                    {p.name} {p.isHost && <span title="Host" className="badge"><Crown className="stat-card-icon" /></span>}
                  </div>
                  <small className="muted">{p.status === 'online' ? 'Online' : 'Idle'}</small>
                </div>
              </div>
              <div style={{ position: 'relative' }}>
                <button className="btn btn-secondary btn-icon" onClick={() => setOpenMenu(openMenu === p.id ? null : p.id)} aria-haspopup="menu" aria-expanded={openMenu === p.id}>
                  <MoreVertical />
                </button>
                {openMenu === p.id && (
                  <div role="menu" className="menu glass-panel p-4" style={{ position: 'absolute', right: 0, top: '110%', minWidth: 220 }}>
                    <button className="btn btn-block" onClick={() => { kickPlayer(p.id); setOpenMenu(null) }}>Kick Player</button>
                    <button className="btn btn-block" onClick={() => { transferHost(p.id); setOpenMenu(null) }}>Transfer Host</button>
                    <button className="btn btn-block" onClick={() => setOpenMenu(null)}>Player List</button>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
