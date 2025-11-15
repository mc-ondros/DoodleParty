const path = require('path');
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
// Existing canvas/static assets
const staticDir = path.join(__dirname, '..', 'public');
const indexPath = path.join(staticDir, 'index.html');
const senderPath = path.join(staticDir, 'drawing_sender.html');
const doodlepartyPath = path.join(staticDir, 'doodleparty.html');

// Admin React build (served from root dist)
const adminDistDir = path.join(__dirname, '..', '..', 'dist');
const adminIndexPath = path.join(adminDistDir, 'index.html');

// Serve canvas/static assets
app.use(express.static(staticDir, {
        maxAge: '1d',
        immutable: true
}));

// Serve admin build assets under /admin (JS, CSS, etc.)
if (require('fs').existsSync(adminDistDir)) {
    app.use('/admin', express.static(adminDistDir, {
        maxAge: '1h'
    }));
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

io.on('connection', (socket) => {
    console.log(`socket.io - client connected (${socket.id})`);
    let heartbeatId = null;

    if (DEMO_MODE) {
        socket.emit('quickdraw.drawing', drawingSample);
        socket.emit('quickdraw.batch', batchSample);

        heartbeatId = setInterval(() => {
            socket.emit('quickdraw.stroke', heartbeatStroke[0]);
        }, 8000);
    }

    socket.on('quickdraw.ack', (payload) => {
        console.log('Received ack from client', payload);
    });

    ['stroke', 'batch', 'drawing', 'clear'].forEach((eventName) => {
        socket.on(`quickdraw.${eventName}`, (payload) => {
            socket.broadcast.emit(`quickdraw.${eventName}`, payload);
            console.log(`Relayed quickdraw.${eventName} from ${socket.id}`);
        });
    });

    socket.on('disconnect', (reason) => {
        if (heartbeatId) {
            clearInterval(heartbeatId);
        }
        console.log(`socket.io - client disconnected (${socket.id}): ${reason}`);
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
