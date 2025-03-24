const socketIO = require('socket.io');

function initializeSocket(server) {
    const io = socketIO(server);

    io.on('connection', (socket) => {
        console.log('클라이언트 연결됨');

        socket.on('disconnect', () => {
            console.log('클라이언트 연결 해제');
        });
    });

    return io;
}

module.exports = initializeSocket;
