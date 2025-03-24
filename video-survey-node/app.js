const express = require('express');
const path = require('path');
const connectDB = require('./config/database');
const fs = require('fs');
const { spawn } = require('child_process');
const app = express();
const port = 3000;
const server = require('http').createServer(app);
const initializeSocket = require('./socket');
const { router: analysisRouter, setIO } = require('./routes/analysis');

const foldersRouter = require('./routes/folders');
const statisticsRouter = require('./routes/statistics');

// 미들웨어 설정
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static('public'));

// uploads 폴더의 정적 파일 서빙 설정 추가
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// EJS 템플릿 엔진 설정
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// 라우트 설정
const indexRouter = require('./routes/index');
const recordingRoutes = require('./routes/recording');
const surveyRoutes = require('./routes/survey');
app.use('/', indexRouter);
app.use('/api', recordingRoutes);
app.use('/api', surveyRoutes);

app.get('/', (req, res) => {
    res.render('index');
});

// 일반 페이지 라우트
app.get('/survey', (req, res) => {
  res.render('survey');
});

// API 라우트 등록 (순서 변경)
app.use('/api', analysisRouter);
app.use('/api', foldersRouter);
app.use('/api', statisticsRouter);

// 에러 처리 미들웨어
app.use((err, req, res, next) => {
    console.error('에러 발생:', err.stack);
    res.status(500).json({
        status: 'error',
        message: err.message || '서버 내부 오류가 발생했습니다.'
    });
});
const uploadRouter = require('./routes/upload');
app.use('/upload', uploadRouter);

// 정적 파일 서빙 설정 추가
app.use('/data', express.static('data'));

// analysis_results 디렉토리 생성 확인
const analysisResultsDir = path.join(__dirname, 'data', 'analysis_results');
if (!fs.existsSync(analysisResultsDir)) {
    fs.mkdirSync(analysisResultsDir, { recursive: true });
}

// uploads 폴더가 없으면 생성
const uploadsDir = path.join(__dirname, 'uploads');
const videosDir = path.join(uploadsDir, 'videos');
if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir);
if (!fs.existsSync(videosDir)) fs.mkdirSync(videosDir);

// API 라우트 추가
const checkResponseSent = (res) => {
    if (res.headersSent) {
        console.warn('Response already sent');
        return true;
    }
    return false;
};

app.post('/api/analyze', express.json(), async (req, res) => {
    try {
        const folder = req.body.folder || req.query.folder;
        if (!folder) {
            return res.status(400).json({ 
                success: false,
                error: '폴더 파라미터가 필요합니다.',
                details: '분석할 폴더를 지정해주세요.'
            });
        }

        const videoDir = path.join(__dirname, 'uploads', 'videos', folder);
        if (!fs.existsSync(videoDir)) {
            return res.status(404).json({
                success: false,
                error: '폴더를 찾을 수 없습니다.',
                details: `'${folder}' 폴더가 존재하지 않습니다.`
            });
        }

        // 즉시 응답 보내기
        res.json({ 
            success: true,
            message: '분석이 시작되었습니다.',
            folder: folder 
        });

        // TensorFlow 경고 메시지 숨기기 위한 환경변수 설정
        const env = {
            ...process.env,
            TF_CPP_MIN_LOG_LEVEL: '3'  // 경고 메시지 숨김
        };
        
        const pythonProcess = spawn('/home/jaeseok/anaconda3/envs/wemeet/bin/python', [
            path.join(__dirname, 'scripts', 'analyze.py'),
            videoDir
        ], { env });  // 환경변수 적용

        // Socket.IO 인스턴스 가져오기
        const io = req.app.get('io');

        // 모든 상태 업데이트는 Socket.IO로 처리
        pythonProcess.stdout.on('data', (data) => {
            try {
                const output = data.toString().trim();
                console.log('Python 출력:', output);
                
                try {
                    const parsed = JSON.parse(output);
                    if (parsed.progress !== undefined) {
                        io.emit('analysisProgress', { progress: parsed.progress });
                    }
                } catch (err) {
                    console.log('일반 출력:', output);
                }
            } catch (err) {
                console.error('Python 출력 처리 오류:', err);
                io.emit('analysisError', { error: err.message });
            }
        });

        pythonProcess.stderr.on('data', (data) => {
            console.error('Python 에러:', data.toString());
            io.emit('analysisError', { message: data.toString() });
        });

        pythonProcess.on('close', (code) => {
            console.log('Python 프로세스 종료. 코드:', code);
            if (code === 0) {
                io.emit('analysisComplete', { status: 'success' });
            } else {
                io.emit('analysisError', { error: '분석 프로세스 실패' });
            }
        });

    } catch (error) {
        console.error('분석 에러:', error);
        // 응답을 아직 보내지 않은 경우에만 에러 응답
        if (!checkResponseSent(res)) {
            return res.status(500).json({
                success: false,
                error: '분석 중 오류가 발생했습니다.',
                details: error.message,
                stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
            });
        }
        // Socket.IO로 에러 전송
        const io = req.app.get('io');
        io.emit('analysisError', { 
            error: '분석 중 오류가 발생했습니다.',
            details: error.message 
        });
    }
});

// 분석 결과 조회 API
app.get('/api/analysis-result', (req, res) => {
    try {
        const { file } = req.query;
        if (!file) {
            return res.status(400).json({ error: '파일명이 지정되지 않았습니다.' });
        }

        const filePath = path.join(__dirname, file);
        if (!fs.existsSync(filePath)) {
            return res.status(404).json({ error: '파일을 찾을 수 없습니다.' });
        }

        const result = JSON.parse(fs.readFileSync(filePath, 'utf8'));
        res.json(result);

    } catch (error) {
        console.error('Result reading error:', error);
        res.status(500).json({ error: error.message });
    }
});

// analysis-report 라우트 수정
app.get('/analysis-report', (req, res) => {
    const folder = req.query.folder;
    console.log('분석 리포트 요청 폴더:', folder);
    
    if (!folder) {
        // 폴더 파라미터가 없을 경우 가장 최근 분석 결과 파일 찾기
        const analysisResultsDir = path.join(__dirname, 'data', 'analysis_results');
        try {
            const files = fs.readdirSync(analysisResultsDir)
                           .filter(file => file.endsWith('.json'))
                           .sort((a, b) => {
                               return fs.statSync(path.join(analysisResultsDir, b)).mtime.getTime() -
                                      fs.statSync(path.join(analysisResultsDir, a)).mtime.getTime();
                           });
            
            if (files.length > 0) {
                // 가장 최근 파일의 이름을 folder 파라미터로 사용
                const latestFile = files[0].replace('.json', '');
                return res.redirect(`/analysis-report?folder=${latestFile}`);
            }
        } catch (error) {
            console.error('최근 분석 파일 검색 실패:', error);
        }
    }
    
    res.render('analysis-report');
});

// analysis 라우터는 나중에 정의
app.use('/analysis', analysisRouter);

// API 라우트
app.get('/api/survey-file', (req, res) => {
    const folders = req.query.folders;
    // 설문 데이터 처리 로직
});

// Socket.IO 인스턴스를 라우터에서 사용할 수 있도록 설정
const io = initializeSocket(server);
app.set('io', io);

// Socket.IO 인스턴스 전달
setIO(io);

// 소켓 연결 처리
io.on('connection', (socket) => {
    console.log('클라이언트 연결됨');
    
    socket.on('disconnect', () => {
        console.log('클라이언트 연결 해제');
    });
});

// MongoDB 연결
connectDB()
  .then(() => {
    console.log('MongoDB가 성공적으로 연결되었습니다.');
    
    // 서버 시작
    server.listen(port, () => {
      console.log(`서버가 http://localhost:${port} 에서 실행 중입니다.`);
    });
  })
  .catch(err => {
    console.error('MongoDB 연결 실패:', err);
    process.exit(1);
  });
