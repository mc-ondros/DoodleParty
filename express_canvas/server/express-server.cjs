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
const staticDir = path.join(__dirname, '..', 'public');
const indexPath = path.join(staticDir, 'index.html');
const senderPath = path.join(staticDir, 'drawing_sender.html');

app.use(express.static(staticDir, {
    maxAge: '1d',
    immutable: true
}));

app.get('/', (req, res) => res.sendFile(indexPath));

app.get('/health', (req, res) => {
    res.json({ status: 'ok', socket: io.engine ? 'ready' : 'down' });
});

app.get('/quickdraw-sender', (req, res) => {
    res.sendFile(senderPath);
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
    console.log(`Express QuickDraw server listening on http://${hostDisplay}:${PORT}`);
});
