import React, { useEffect, useState } from 'react';

import { Clock, Crown, Lock, MoreVertical, Users } from 'lucide-react';

/**
 * Admin Dashboard Component
 *
 * Administrative interface for managing game settings,
 * player list, moderation, and session configuration.
 *
 * Related:
 * - src/hooks/useDemoTimer.tsx (timer functionality)
 * - public/css/styles/dashboard.css (dashboard styling)
 *
 * Exports:
 * - AdminPanel (default export)
 */

// Constants
const DEFAULT_GAME_DURATION = 300; // 5 minutes in seconds
const DEFAULT_TIMER_PRESET = '300'; // 5 minutes
const DEFAULT_CUSTOM_TIMER = 90; // 1.5 minutes in seconds
const MILLISECONDS_PER_SECOND = 1000;

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
    }, MILLISECONDS_PER_SECOND)
    return () => clearInterval(id)
  }, [state])
  const minutes = Math.floor(seconds / 60)
  const rem = seconds % 60
  return { minutes, rem, state, setState, reset: (s = initialSeconds) => { setSeconds(s); setState('running') } }
}

export default function AdminPanel() {
  const { minutes, rem, state, setState, reset } = useDemoTimer(DEFAULT_GAME_DURATION)
  const [duration, setDuration] = useState(DEFAULT_GAME_DURATION)

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

  const [customTimer, setCustomTimer] = useState(DEFAULT_CUSTOM_TIMER)
  const [timerPreset, setTimerPreset] = useState<string>(DEFAULT_TIMER_PRESET)

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
      {/* Header */}
      <div className="doodle-header"><h1>Doodle Party</h1></div>

      {/* Top row: Time + Game Mode */}
      <div className="row row-top" style={{ gridColumn: '1 / -1' }}>
        <div className="glass-panel p-6">
          <h2 className="panel-title"><Clock style={{ width: 28, height: 28 }} /> Time:</h2>
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
                <input type="number" min={10} step={5} value={customTimer}
                  onChange={(e) => { const v = Math.max(10, Number(e.target.value)||10); setCustomTimer(v); setConfig(c=>({ ...c, roundTimer: v })); setDuration(v); reset(v) }} />
              )}
            </div>
          </div>
          <div className="timer-value">{minutes}:{String(rem).padStart(2, '0')}</div>
          <div className="form-group-inline">
            <button className="btn btn-primary" onClick={() => setState('running')}>start</button>
            <button className="btn btn-secondary" onClick={() => setState('paused')}>pause</button>
            <button className="btn btn-secondary" onClick={() => reset(duration)}>reset</button>
            <button className="btn btn-danger" onClick={() => setState('expired')}>stop</button>
          </div>
        </div>

        <div className="glass-panel p-6">
          <h2 className="panel-title">Game Mode:</h2>
          <div className="mode-chip-group">
            {['Classic','Speed','Story'].map(label => (
              <button key={label}
                className={`btn mode-btn ${
                  (label==='Classic' && config.gameMode==='Classic') ||
                  (label==='Speed' && (config.gameMode==='Speed' || config.gameMode==='Speed Sketch')) ||
                  (label==='Story' && (config.gameMode==='Story' || config.gameMode==='Story Canvas'))
                    ? 'active' : ''}`}
                onClick={() => setConfig(c => ({ ...c, gameMode: label === 'Speed' ? 'Speed' : label }))}
              >{label}</button>
            ))}
          </div>
        </div>
      </div>

      {/* Row 2: Game Config + Prompt */}
      <div className="row row-2" style={{ gridColumn: '1 / -1' }}>
        <div className="glass-panel p-6">
          <h2 className="panel-title">Game Configuration</h2>
          <div className="form-group">
            <label>set max players</label>
            <select value={String(config.maxPlayers)} onChange={(e) => setConfig(c => ({ ...c, maxPlayers: Number(e.target.value) }))}>
              <option value={4}>4</option>
              <option value={8}>8</option>
              <option value={16}>16</option>
              <option value={32}>32</option>
              <option value={-1}>unlimited</option>
            </select>
          </div>
          <div className="form-group">
            <label>set ink limit (Low, Medium, High, Unlimited)</label>
            <select value={config.inkLimit} onChange={(e) => setConfig(c => ({ ...c, inkLimit: e.target.value as any }))}>
              <option>Low</option>
              <option>Medium</option>
              <option>High</option>
              <option>Unlimited</option>
            </select>
          </div>
          <div className="form-group-inline">
            <div className="form-group">
              <label>Enable/ disable teams</label>
              <select value={String(config.teamsEnabled)} onChange={(e) => setConfig(c => ({ ...c, teamsEnabled: e.target.value === 'true' }))}>
                <option value={'false'}>disabled</option>
                <option value={'true'}>enabled</option>
              </select>
            </div>
            <div className="form-group">
              <label>Private vs Public</label>
              <select value={config.visibility} onChange={(e) => setConfig(c => ({ ...c, visibility: e.target.value as any }))}>
                <option>Public</option>
                <option>Private</option>
              </select>
            </div>
          </div>
          {config.visibility === 'Private' && (
            <div className="form-group">
              <label>Password-protected</label>
              <input type="password" value={config.password} onChange={(e) => setConfig(c => ({ ...c, password: e.target.value }))} placeholder="enter password" />
            </div>
          )}
        </div>

        <div className="glass-panel p-6">
          <h2 className="panel-title">Custom prompt for drawing</h2>
          <textarea placeholder="write here"
            value={config.customPrompt}
            onChange={(e) => setConfig(c => ({ ...c, customPrompt: e.target.value }))}
          />
          <div className="form-group-inline" style={{ marginTop: 12 }}>
            <div className="form-group">
              <label>Difficulty</label>
              <select value={config.difficulty} onChange={(e) => setConfig(c => ({ ...c, difficulty: e.target.value as any }))}>
                <option>Easy</option>
                <option>Medium</option>
                <option>Hard</option>
              </select>
            </div>
            <div className="form-group">
              <label>Theme/Category</label>
              <select value={config.category} onChange={(e) => setConfig(c => ({ ...c, category: e.target.value }))}>
                <option>Animals</option>
                <option>Food</option>
                <option>Sports</option>
                <option>Random</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Moderation Settings */}
      <div className="glass-panel p-6" style={{ gridColumn: '1 / -1' }}>
        <h2 className="panel-title">Moderation Settings</h2>
        <div className="form-group-inline">
          <div className="form-group">
            <label>AI Strictness</label>
            <select value={config.aiStrictness} onChange={(e) => setConfig(c => ({ ...c, aiStrictness: e.target.value as any }))}>
              <option>Relaxed</option>
              <option>Normal</option>
              <option>Strict</option>
            </select>
          </div>
          <div className="form-group">
            <label>Auto-kick after violations</label>
            <select value={String(config.autoKick)} onChange={(e) => setConfig(c => ({ ...c, autoKick: Number(e.target.value) }))}>
              <option value={-1}>disabled</option>
              <option value={3}>3</option>
              <option value={5}>5</option>
              <option value={10}>10</option>
            </select>
          </div>
        </div>
      </div>

      {/* Players list */}
      <div className="glass-panel p-6" style={{ gridColumn: '1 / -1' }}>
        <div className="flex items-center justify-between mb-3">
          <h3 className="panel-title" style={{ marginBottom: 0 }}><span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}><Users style={{ width: 24, height: 24 }} /> Player list:</span> {sessionLocked && <span className="badge" title="Session is locked" style={{ marginLeft: 10 }}>Locked</span>}</h3>
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
            <div key={p.id} className="flex items-center justify-between border rounded p-3 mb-2" style={{ borderColor: 'rgba(255,255,255,0.6)' }}>
              <div className="flex items-center gap-3">
                <span className={`status-dot ${p.status}`} aria-label={p.status} title={p.status}></span>
                <div>
                  <div className="flex items-center gap-2" style={{ fontWeight: 400 }}>
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
        <div className="form-group-inline" style={{ marginTop: 12 }}>
          <button
            className={`btn ${sessionLocked ? 'btn-secondary' : 'btn-secondary'}`}
            onClick={() => setSessionLocked((v) => !v)}
            title={sessionLocked ? 'Unlock to allow new players' : 'Lock to prevent new joins'}
          >
            <Lock style={{ width: 20, height: 20 }} /> {sessionLocked ? 'Unlock Session' : 'Lock Session'}
          </button>
        </div>
      </div>
    </div>
  )
}
