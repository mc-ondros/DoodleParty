const path = require('path');
const fs = require('fs');
const fsp = require('fs/promises');
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
    cors: {
        origin: true,
        methods: ['GET', 'POST']
    }
});

const PORT = process.env.PORT || 3000;
const HOST = process.env.HOST || '0.0.0.0';
const DEMO_MODE = process.env.DEMO_MODE === '1';
const ADMIN_USER = process.env.ADMIN_USER || 'admin';
const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || '';
// Existing canvas/static assets
const staticDir = path.join(__dirname, '..', 'public');
const indexPath = path.join(staticDir, 'index.html');
const senderPath = path.join(staticDir, 'drawing_sender.html');
const doodlepartyPath = path.join(staticDir, 'doodleparty.html');

// Admin React build (served from root dist)
const adminDistDir = path.join(__dirname, '..', '..', 'dist');
const adminIndexPath = path.join(adminDistDir, 'index.html');

// Admin config file (repo root)
const adminConfigPath = path.join(__dirname, '..', '..', 'AdminConfig.json');

// Helpers: read/write AdminConfig.json atomically
async function readAdminConfig() {
    try {
        const raw = await fsp.readFile(adminConfigPath, 'utf-8');
        return JSON.parse(raw);
    } catch (err) {
        if (err.code === 'ENOENT') {
            // Seed with defaults if missing
            const defaults = {
                Timer: 300,
                TimerPreset: '300',
                'Game Mode': 'Speed',
                'Max Players': 8,
                'Ink Limit': 'Medium',
                Teams: 'disabled',
                Visibility: 'Public',
                Password: '',
                'Custom Prompt': '',
                'Content Mode': 'SFW',
                Session: 'Open'
            };
            await writeAdminConfig(defaults);
            return defaults;
        }
        throw err;
    }
}

async function writeAdminConfig(data) {
    const dir = path.dirname(adminConfigPath);
    const tmpPath = path.join(dir, `.AdminConfig.json.tmp-${Date.now()}`);
    const json = JSON.stringify(data, null, 2) + '\n';
    await fsp.writeFile(tmpPath, json, { encoding: 'utf-8' });
    await fsp.rename(tmpPath, adminConfigPath);
}

// --- Simple session id (ephemeral on boot) ---
function makeSessionId() {
    const alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
    let out = 'DP-';
    for (let i = 0; i < 4; i++) out += alphabet[Math.floor(Math.random() * alphabet.length)];
    return out;
}
const SESSION_ID = makeSessionId();

// --- Players tracking ---
const players = new Map(); // id -> { id, address }

function emitPlayersUpdate() {
    io.emit('players:update', { count: players.size });
}

// --- Timer (server-authoritative) ---
const timer = {
    state: 'paused', // 'running' | 'paused' | 'expired'
    duration: 300,   // seconds
    remaining: 300,
    _interval: null,
};

function emitTimerUpdate() {
    io.emit('timer:update', { state: timer.state, remaining: timer.remaining, duration: timer.duration });
}

function startTimer() {
    if (timer._interval) clearInterval(timer._interval);
    timer.state = 'running';
    timer._interval = setInterval(() => {
        timer.remaining = Math.max(0, timer.remaining - 1);
        emitTimerUpdate();
        if (timer.remaining <= 0) {
            clearInterval(timer._interval);
            timer._interval = null;
            timer.state = 'expired';
            emitTimerUpdate();
        }
    }, 1000);
    emitTimerUpdate();
}

function pauseTimer() {
    if (timer._interval) {
        clearInterval(timer._interval);
        timer._interval = null;
    }
    timer.state = 'paused';
    emitTimerUpdate();
}

function resetTimer(seconds) {
    const s = Number.isFinite(Number(seconds)) && Number(seconds) > 0 ? Number(seconds) : timer.duration;
    timer.duration = s;
    timer.remaining = s;
    timer.state = 'paused';
    if (timer._interval) {
        clearInterval(timer._interval);
        timer._interval = null;
    }
    emitTimerUpdate();
}

function stopTimer() {
    if (timer._interval) {
        clearInterval(timer._interval);
        timer._interval = null;
    }
    timer.state = 'expired';
    timer.remaining = 0;
    emitTimerUpdate();
}

function getTimerSnapshot() {
    return { state: timer.state, remaining: timer.remaining, duration: timer.duration };
}

// Serve canvas/static assets
app.use(express.json({ limit: '256kb' }));

app.use(express.static(staticDir, {
        maxAge: '1d',
        immutable: true
}));

// Serve admin build assets under /admin (JS, CSS, etc.)
if (require('fs').existsSync(adminDistDir)) {
    // Optional Basic Auth for /admin and admin APIs
    const authEnabled = ADMIN_PASSWORD && ADMIN_PASSWORD.length > 0;
    const basicAuth = (req, res, next) => {
        if (!authEnabled) return next();
        const header = req.headers['authorization'] || '';
        if (!header.startsWith('Basic ')) {
            res.set('WWW-Authenticate', 'Basic realm="DoodleParty Admin"');
            return res.status(401).send('Authentication required');
        }
        const b64 = header.slice(6);
        let userpass = '';
        try { userpass = Buffer.from(b64, 'base64').toString('utf8'); } catch (_) {}
        const idx = userpass.indexOf(':');
        const user = idx >= 0 ? userpass.slice(0, idx) : '';
        const pass = idx >= 0 ? userpass.slice(idx + 1) : '';
        if (user === ADMIN_USER && pass === ADMIN_PASSWORD) return next();
        res.set('WWW-Authenticate', 'Basic realm="DoodleParty Admin"');
        return res.status(401).send('Invalid credentials');
    };

    app.use('/admin', basicAuth, express.static(adminDistDir, {
        maxAge: '1h'
    }));
    // Protect admin APIs too
    const adminApiProtected = ['/api/admin-config', '/api/timer', '/api/players/kick'];
    app.use(adminApiProtected, basicAuth);
}

app.get('/', (req, res) => res.sendFile(indexPath));

app.get('/health', (req, res) => {
    res.json({ status: 'ok', socket: io.engine ? 'ready' : 'down' });
});

app.get('/quickdraw-sender', (req, res) => {
    res.sendFile(senderPath);
});

app.get('/doodleparty', (req, res) => {
    res.sendFile(doodlepartyPath);
});

// Admin config API
app.get('/api/admin-config', async (req, res) => {
    try {
        const cfg = await readAdminConfig();
        res.json(cfg);
    } catch (err) {
        console.error('Failed to read AdminConfig.json:', err.code || err.message);
        res.status(500).json({ error: 'Failed to read AdminConfig.json', code: err.code || 'READ_ERROR' });
    }
});

app.post('/api/admin-config', async (req, res) => {
    try {
        const incoming = req.body || {};
        // Read current config, merge shallowly with incoming known keys only
        const current = await readAdminConfig();
        const allowedKeys = [
            'Timer',
            'TimerPreset',
            'Game Mode',
            'Max Players',
            'Ink Limit',
            'Teams',
            'Visibility',
            'Password',
            'Custom Prompt',
            'Content Mode',
            'Session'
        ];
        const next = { ...current };
        for (const k of allowedKeys) {
            if (Object.prototype.hasOwnProperty.call(incoming, k)) {
                next[k] = incoming[k];
            }
        }

    await writeAdminConfig(next);
    
    // Sync timer duration if Timer changed
    const newTimerValue = Number(next['Timer']);
    if (Number.isFinite(newTimerValue) && newTimerValue > 0 && newTimerValue !== timer.duration) {
        timer.duration = newTimerValue;
        // If timer is paused or expired, also update remaining
        if (timer.state !== 'running') {
            timer.remaining = newTimerValue;
        }
        console.log(`Timer duration synced to ${newTimerValue}s from admin config`);
    }
    
    io.emit('admin-config:update', next);
    io.emit('config:update', next); // alias event for UIs
        res.json({ ok: true });
    } catch (err) {
        const code = err && err.code ? err.code : 'WRITE_ERROR';
        console.error('Failed to write AdminConfig.json:', code, err.message);
        if (code === 'EACCES' || code === 'EPERM') {
            return res.status(403).json({ error: 'Permission denied writing AdminConfig.json. See README AdminConfig section.', code });
        }
        res.status(500).json({ error: 'Failed to write AdminConfig.json', code });
    }
});

// Players count API
app.get('/api/players/count', (req, res) => {
    try {
        const count = io.sockets.sockets.size;
        res.json({ count });
    } catch (err) {
        console.error('Failed to get players count:', err.message);
        res.status(500).json({ error: 'Failed to get players count' });
    }
});

// Players list (optional for future UI)
app.get('/api/players', (req, res) => {
    const list = Array.from(players.values());
    res.json({ count: players.size, list });
});

// Kick a player by socket id
app.post('/api/players/kick', (req, res) => {
    try {
        const { id } = req.body || {};
        if (!id) return res.status(400).json({ error: 'Missing player id' });
        const sock = io.sockets.sockets.get(id);
        if (!sock) return res.status(404).json({ error: 'Player not found' });
        sock.emit('kicked', { reason: 'removed_by_admin' });
        sock.disconnect(true);
        return res.json({ ok: true });
    } catch (err) {
        console.error('Kick failed:', err.message);
        res.status(500).json({ error: 'Kick failed' });
    }
});

// Timer control (server-authoritative)
app.post('/api/timer', (req, res) => {
    try {
        const { action, seconds } = req.body || {};
        switch (action) {
            case 'start':
                if (timer.state !== 'running' && timer.remaining <= 0) {
                    // If expired, reset to duration before starting
                    resetTimer(timer.duration);
                }
                startTimer();
                return res.json({ ok: true });
            case 'pause':
                pauseTimer();
                return res.json({ ok: true });
            case 'reset':
                resetTimer(seconds);
                return res.json({ ok: true });
            case 'stop':
                stopTimer();
                return res.json({ ok: true });
            default:
                return res.status(400).json({ error: 'Invalid action' });
        }
    } catch (err) {
        console.error('Timer action failed:', err.message);
        res.status(500).json({ error: 'Timer action failed' });
    }
});

// Combined state for bootstrapping clients
app.get('/api/state', async (req, res) => {
    try {
        const cfg = await readAdminConfig();
        res.json({
            sessionId: SESSION_ID,
            config: cfg,
            timer: getTimerSnapshot(),
            players: { count: players.size }
        });
    } catch (err) {
        console.error('Failed to get state:', err.message);
        res.status(500).json({ error: 'Failed to get state' });
    }
});

// Timer snapshot
app.get('/api/timer', (req, res) => {
    res.json(getTimerSnapshot());
});

// Admin SPA fallback routes
app.get(['/admin', '/admin/*'], (req, res) => {
    if (require('fs').existsSync(adminIndexPath)) {
        res.sendFile(adminIndexPath);
    } else {
        res.status(503).send('Admin UI not built yet. Run: npm run build');
    }
});

const drawingSample = [
    [
        [20, 60, 120, 180, 230, 250],
        [230, 190, 140, 110, 80, 60],
        [0, 120, 210, 320, 420, 510]
    ],
    [
        [140, 160, 180, 200, 220],
        [140, 110, 90, 80, 85],
        [0, 100, 200, 300, 400]
    ]
];

const batchSample = [
    [
        [35, 120, 200, 255, 230, 180],
        [250, 200, 160, 140, 160, 200],
        [0, 140, 230, 320, 410, 500]
    ],
    [
        [68, 80, 110, 150, 190, 210],
        [60, 80, 100, 90, 70, 55],
        [0, 90, 180, 260, 340, 430]
    ]
];

const heartbeatStroke = [
    [
        [120, 130, 140, 150],
        [50, 70, 90, 110],
        [0, 120, 240, 360]
    ]
];

// Store all strokes for syncing new clients
const strokeHistory = [];
const MAX_HISTORY = 5000; // Limit history to prevent memory issues

io.on('connection', (socket) => {
    console.log(`socket.io - client connected (${socket.id})`);
    const address = (socket.handshake && socket.handshake.address) || 'unknown';
    players.set(socket.id, { id: socket.id, address });
    emitPlayersUpdate();
    // Send initial state snapshot to the newly connected client
    readAdminConfig()
        .then((cfg) => {
            socket.emit('state:init', {
                sessionId: SESSION_ID,
                config: cfg,
                timer: getTimerSnapshot(),
                players: { count: players.size },
            });
        })
        .catch((err) => {
            console.error('state:init readAdminConfig failed:', err.message);
        });
    let heartbeatId = null;

    if (DEMO_MODE) {
        socket.emit('quickdraw.drawing', drawingSample);
        socket.emit('quickdraw.batch', batchSample);

        heartbeatId = setInterval(() => {
            socket.emit('quickdraw.stroke', heartbeatStroke[0]);
        }, 8000);
    }

    // Send existing strokes to new client
    if (strokeHistory.length > 0) {
        console.log(`Sending ${strokeHistory.length} strokes to new client ${socket.id}`);
        socket.emit('quickdraw.sync', strokeHistory);
    }

    socket.on('quickdraw.ack', (payload) => {
        console.log('Received ack from client', payload);
    });

    socket.on('quickdraw.requestSync', () => {
        console.log(`Client ${socket.id} requested sync, sending ${strokeHistory.length} strokes`);
        socket.emit('quickdraw.sync', strokeHistory);
    });

    socket.on('quickdraw.stroke', (payload) => {
        socket.broadcast.emit('quickdraw.stroke', payload);
        // Store stroke in history
        strokeHistory.push(payload);
        if (strokeHistory.length > MAX_HISTORY) {
            strokeHistory.shift(); // Remove oldest stroke
        }
        console.log(`Relayed quickdraw.stroke from ${socket.id} (history: ${strokeHistory.length})`);
    });

    socket.on('quickdraw.batch', (payload) => {
        socket.broadcast.emit('quickdraw.batch', payload);
        // Add batch strokes to history
        if (Array.isArray(payload)) {
            payload.forEach(stroke => {
                strokeHistory.push(stroke);
            });
            // Trim if needed
            while (strokeHistory.length > MAX_HISTORY) {
                strokeHistory.shift();
            }
        }
        console.log(`Relayed quickdraw.batch from ${socket.id} (history: ${strokeHistory.length})`);
    });

    socket.on('quickdraw.drawing', (payload) => {
        socket.broadcast.emit('quickdraw.drawing', payload);
        // Add drawing strokes to history
        if (Array.isArray(payload)) {
            payload.forEach(stroke => {
                strokeHistory.push(stroke);
            });
            while (strokeHistory.length > MAX_HISTORY) {
                strokeHistory.shift();
            }
        }
        console.log(`Relayed quickdraw.drawing from ${socket.id} (history: ${strokeHistory.length})`);
    });

    socket.on('quickdraw.clear', (payload) => {
        // Broadcast to ALL clients including sender
        io.emit('quickdraw.clear', payload);
        // Clear history when canvas is cleared
        strokeHistory.length = 0;
        console.log(`Relayed quickdraw.clear from ${socket.id} to all canvas clients, history cleared`);
    });

    socket.on('ml.detectObjects', (payload) => {
        console.log(`Received ML object detection from ${socket.id}:`, {
            sessionId: payload.sessionId,
            objectCount: payload.objects?.length || 0,
            timestamp: payload.timestamp
        });
        
        // Log object details
        if (payload.objects && Array.isArray(payload.objects)) {
            payload.objects.forEach((obj, idx) => {
                console.log(`  Object ${idx}: bbox=(${obj.boundingBox.x1},${obj.boundingBox.y1}) to (${obj.boundingBox.x2},${obj.boundingBox.y2})`);
            });
        }
        
        // Save images to disk for training data
        if (payload.objects && payload.objects.length > 0) {
            const dataDir = path.join(__dirname, '..', '..', 'data', 'ml_detections');
            const sessionDir = path.join(dataDir, payload.sessionId);
            
            // Create directories if they don't exist
            fs.mkdirSync(sessionDir, { recursive: true });
            
            payload.objects.forEach((obj, idx) => {
                const base64Data = obj.image.replace(/^data:image\/png;base64,/, '');
                const filename = `object_${idx}_${payload.timestamp}.png`;
                const filepath = path.join(sessionDir, filename);
                
                fs.writeFile(filepath, base64Data, 'base64', (err) => {
                    if (err) {
                        console.error(`Error saving object ${idx}:`, err);
                    } else {
                        console.log(`💾 Saved object ${idx} to ${filepath}`);
                    }
                });
            });
        }
        
        // Forward to ML server for inference
        console.log('📤 Forwarding to ML server for inference...');
        io.emit('ml.detectObjects', payload);
        
        // Send acknowledgment back to client
        socket.emit('ml.detectObjectsAck', {
            success: true,
            objectCount: payload.objects?.length || 0,
            sessionId: payload.sessionId,
            message: 'Forwarded to ML server'
        });
    });
    
    // Relay ML results back to clients
    socket.on('ml.detectionResults', (results) => {
        console.log('🤖 Received ML results:', {
            sessionId: results.sessionId,
            success: results.success,
            summary: results.summary
        });
        
        // Broadcast results to all clients
        io.emit('ml.detectionResults', results);
    });
    
    // Handle region erasure (for inappropriate content removal)
    socket.on('quickdraw.eraseRegion', (payload) => {
        const { x1, y1, x2, y2, reason } = payload;
        console.log('');
        console.log('═'.repeat(70));
        console.log('🚨 INAPPROPRIATE CONTENT REMOVAL');
        console.log('═'.repeat(70));
        console.log(`Source: ${socket.id}`);
        console.log(`Region: (${x1}, ${y1}) → (${x2}, ${y2})`);
        console.log(`Size: ${x2-x1}×${y2-y1} pixels`);
        console.log(`Reason: ${reason || 'ML detection - inappropriate content'}`);
        console.log(`Action: Broadcasting removal to all clients`);
        console.log('═'.repeat(70));
        console.log('');
        
        // Broadcast to all clients including sender for sync
        io.emit('quickdraw.eraseRegion', payload);
    });

    socket.on('disconnect', (reason) => {
        players.delete(socket.id);
        emitPlayersUpdate();
        if (heartbeatId) {
            clearInterval(heartbeatId);
        }
        console.log(`socket.io - client disconnected (${socket.id}), reason: ${reason}`);
    });
});

server.listen(PORT, HOST, () => {
    const hostDisplay = HOST === '0.0.0.0' ? '0.0.0.0 (all interfaces)' : HOST;
    console.log(`Express server listening on http://${hostDisplay}:${PORT}`);
    console.log('');
    console.log('Access from WSL/localhost:');
    console.log(`  http://localhost:${PORT}`);
    console.log(`  http://localhost:${PORT}/doodleparty`);
    console.log(`  http://localhost:${PORT}/admin`);
    console.log('');
    
    // Get local network IP addresses
    const os = require('os');
    const networkInterfaces = os.networkInterfaces();
    const addresses = [];
    const wslAddresses = [];
    
    Object.keys(networkInterfaces).forEach(interfaceName => {
        networkInterfaces[interfaceName].forEach(iface => {
            if (iface.family === 'IPv4' && !iface.internal) {
                addresses.push(iface.address);
                // WSL2 typically uses eth0 interface
                if (interfaceName.toLowerCase().includes('eth')) {
                    wslAddresses.push(iface.address);
                }
            }
        });
    });
    
    if (wslAddresses.length > 0) {
        console.log('Access from Windows host (WSL bridge):');
        wslAddresses.forEach(addr => {
            console.log(`  http://${addr}:${PORT}`);
            console.log(`  http://${addr}:${PORT}/doodleparty`);
            console.log(`  http://${addr}:${PORT}/admin`);
        });
        console.log('');
    }
    
    if (addresses.length > 0) {
        console.log('Access from mobile/other devices on same network:');
        addresses.forEach(addr => {
            console.log(`  http://${addr}:${PORT}`);
            console.log(`  http://${addr}:${PORT}/doodleparty`);
            console.log(`  http://${addr}:${PORT}/admin`);
        });
        console.log('');
        console.log('Note: If running in WSL2, you may need to:');
        console.log('1. Allow port through Windows Firewall');
        console.log('2. Use Windows IP address (run "ipconfig" in Windows cmd)');
        console.log('3. Or set up port forwarding: netsh interface portproxy add v4tov4 listenport=3000 listenaddress=0.0.0.0 connectport=3000 connectaddress=<WSL_IP>');
        console.log('');
    }
});
