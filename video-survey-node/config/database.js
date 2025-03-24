const mongoose = require('mongoose');

const connectDB = async () => {
    try {
        const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/video-survey';
        await mongoose.connect(MONGODB_URI, {
            useNewUrlParser: true,
            useUnifiedTopology: true
        });
        console.log('MongoDB 연결 성공');
    } catch (err) {
        console.error('MongoDB 연결 실패:', err);
        process.exit(1);
    }
};

module.exports = connectDB;