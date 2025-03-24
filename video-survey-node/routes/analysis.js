/**
 * analysis.js
 * - 분석과 관련된 서버 라우트들을 관리하는 파일
 * - 중복된 router.get('/analysis-result', ...)와 router.get('/api/analysis-result', ...)를 정리하였음
 */

const express = require('express');
const router = express.Router();
const path = require('path');
const fs = require('fs');
const fsPromises = fs.promises;
const { spawn } = require('child_process');

// Socket.IO 인스턴스를 저장할 변수
let io;

// Socket.IO 설정 함수 추가
const setIO = (_io) => {
    io = _io;
};

// 분석 진행 상태를 저장할 Map
const analysisProgressMap = new Map();

// 응답 상태 체크 유틸리티 함수 추가
const checkResponseSent = (res) => {
    if (res.headersSent) {
        return true;
    }
    return false;
};

// ----------------------------------------------------
// 1. 썸네일 생성
router.post('/create-thumbnails', async (req, res) => {
    /**
     * folders에 포함된 각 폴더마다 파이썬 스크립트를 실행하여 썸네일을 생성함
     */
    const { folders } = req.body;
    try {
        console.log('처리할 폴더들:', folders);

        for (const folder of folders) {
            const folderPath = path.join(__dirname, '..', 'uploads', 'videos', folder);
            const thumbnailPath = path.join(folderPath, 'thumbnail.jpg');
            
            if (!fs.existsSync(folderPath)) {
                console.error(`폴더가 존재하지 않음: ${folderPath}`);
                continue;
            }

            const files = await fsPromises.readdir(folderPath);
            const videoFiles = files.filter(file => file.endsWith('.mp4'));
            
            if (videoFiles.length === 0) {
                console.error(`${folder}에 비디오(.mp4) 파일이 없습니다.`);
                continue;
            }

            const videoPath = path.join(folderPath, videoFiles[0]);
            console.log(`썸네일 생성 시도: ${thumbnailPath}`);

            try {
                const pythonProcess = spawn('python3', [
                    path.join(__dirname, '..', 'scripts', 'create_thumbnail.py'),
                    videoPath,
                    thumbnailPath
                ]);

                await new Promise((resolve, reject) => {
                    pythonProcess.on('close', (code) => {
                        if (code === 0) {
                            resolve();
                        } else {
                            reject(new Error(`썸네일 생성 실패 (종료코드: ${code})`));
                        }
                    });
                });

            } catch (err) {
                console.error(`${folder} 처리 중 에러:`, err);
                continue;
            }
        }
        res.json({ status: 'success' });
    } catch (err) {
        console.error('전체 처리 실패:', err);
        res.status(500).json({ error: '썸네일 생성에 실패했습니다.' });
    }
});

// ----------------------------------------------------
// 2. 분석 시작
router.post('/start-analysis', async (req, res) => {
    const { folders } = req.body;
    const io = req.app.get('io');
    
    if (!folders || !Array.isArray(folders) || folders.length === 0) {
        return res.status(400).json({ error: '유효한 폴더 목록이 필요합니다.' });
    }
    
    try {
        // 진행 상태 초기화
        folders.forEach(folder => {
            analysisProgressMap.set(folder, 0);
        });

        // 즉시 응답 보내기
        res.json({ status: 'success' });

        const videoDir = path.join(__dirname, '..', 'uploads', 'videos', folders[0]);
        // Anaconda 가상환경의 Python 실행 경로 설정
        const pythonPath = '/home/jaeseok/anaconda3/envs/wemeet/bin/python';  
        const pythonProcess = spawn(pythonPath, [
            path.join(__dirname, '..', 'scripts', 'analyze.py'),
            videoDir
        ]);

        // Python 스크립트 출력 처리
        pythonProcess.stdout.on('data', (data) => {
            try {
                const lines = data.toString().trim().split('\n');
                for (const line of lines) {
                    try {
                        const parsed = JSON.parse(line);
                        if (parsed.type === 'status') {
                            // update_status 메시지만 전달
                            io.emit('analysisStatus', { message: parsed.message });
                        }
                    } catch (err) {
                        // JSON이 아닌 출력은 무시
                    }
                }
            } catch (err) {
                // 출력 처리 오류는 콘솔에만 기록
                console.error('Python 출력 처리 오류:', err);
            }
        });

        // 프로세스 종료 처리 - 성공 시에만 메시지 전송
        pythonProcess.on('close', (code) => {
            if (code === 0) {
                analysisProgressMap.set(folders[0], 100);
                io.emit('analysisComplete', { status: 'success' });
            }
        });

    } catch (err) {
        // 응답을 아직 보내지 않은 경우에만 에러 응답
        if (!checkResponseSent(res)) {
            console.error('분석 시작 실패:', err);
            res.status(500).json({ error: err.message });
        }
        // 소켓으로는 항상 에러 전송
        io.emit('analysisError', { error: err.message });
    }
});

// ----------------------------------------------------
// 4. 썸네일 정리(삭제)
router.post('/cleanup-thumbnails', async (req, res) => {
    /**
     * folders 배열에 해당하는 모든 폴더들 내의 thumbnail.jpg를 제거
     */
    const { folders } = req.body;
    
    try {
        for (const folder of folders) {
            const thumbnailPath = path.join(__dirname, '..', 'uploads', 'videos', folder, 'thumbnail.jpg');
            if (fs.existsSync(thumbnailPath)) {
                fs.unlinkSync(thumbnailPath);
            }
        }
        res.json({ status: 'success' });
    } catch (err) {
        console.error('썸네일 삭제 실패:', err);
        res.status(500).json({ error: '썸네일 삭제에 실패했습니다.' });
    }
});

// ----------------------------------------------------
// 5. 분석 보고서 페이지 렌더링
router.get('/report', (req, res) => {
    const folders = req.query.folders;
    if (folders) {
        res.redirect(`/analysis-report?folder=${encodeURIComponent(folders)}`);
    } else {
        res.redirect('/analysis-report');
    }
});

// ----------------------------------------------------
// 6. 비디오 스트리밍 라우트
router.get('/videos/:videoPath', (req, res) => {
    /**
     * :videoPath를 이용해 /uploads/videos/... 에 있는 mp4 파일을 스트리밍
     */
    const videoPath = path.join(process.cwd(), 'uploads', 'videos', req.params.videoPath);
    
    if (!fs.existsSync(videoPath)) {
        return res.status(404).send('Video not found');
    }

    const stat = fs.statSync(videoPath);
    const fileSize = stat.size;
    const range = req.headers.range;

    if (range) {
        const parts = range.replace(/bytes=/, "").split("-");
        const start = parseInt(parts[0], 10);
        const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
        const chunksize = (end - start) + 1;
        const file = fs.createReadStream(videoPath, { start, end });
        const head = {
            'Content-Range': `bytes ${start}-${end}/${fileSize}`,
            'Accept-Ranges': 'bytes',
            'Content-Length': chunksize,
            'Content-Type': 'video/mp4',
        };
        res.writeHead(206, head);
        file.pipe(res);
    } else {
        const head = {
            'Content-Length': fileSize,
            'Content-Type': 'video/mp4',
        };
        res.writeHead(200, head);
        fs.createReadStream(videoPath).pipe(res);
    }
});

// ----------------------------------------------------
// 7. 감정 분석 결과 API
router.get('/api/emotion-analysis', async (req, res) => {
    /**
     * analysis-report.js(프론트)에서 fetch('/analysis/api/emotion-analysis') 로 호출
     * 가장 최신의 분석 결과 JSON 파일을 찾아서 응답
     */
    try {
        const analysisDir = path.join(__dirname, '..', 'data', 'analysis_results');
        
        const files = await fsPromises.readdir(analysisDir);
        const jsonFiles = files.filter(file => file.endsWith('.json'));
        
        if (jsonFiles.length === 0) {
            console.error('분석 결과 파일이 없음');
            return res.status(404).json({ error: '분석 결과를 찾을 수 없습니다.' });
        }

        // 최신 파일(가장 나중에 정렬된 파일)
        const latestFile = jsonFiles.sort().reverse()[0];
        const filePath = path.join(analysisDir, latestFile);
        
        console.log('최신 분석 파일:', filePath); // 디버깅용
        
        const data = await fsPromises.readFile(filePath, 'utf8');
        const analysisData = JSON.parse(data);
        
        console.log('데이터 로드 성공(키들):', Object.keys(analysisData)); // 디버깅용
        res.json(analysisData);
    } catch (error) {
        console.error('분석 데이터 로딩 에러:', error);
        res.status(500).json({ 
            error: '분석 데이터를 불러오는데 실패했습니다.',
            details: error.message 
        });
    }
});

// ----------------------------------------------------
// 8. 기타: 설문 통계 등 추가로 필요한 경우
// 필요 시 /api/analysis-result 등 다른 API를 여기에 함께 두되
// 중복되지 않도록 관리하면 됨.

// 분석 진행 상태 확인 엔드포인트
router.get('/api/progress', (req, res) => {
    const { folders } = req.query;
    if (!folders) {
        return res.status(400).json({ error: '폴더 파라미터가 필요합니다.' });
    }
    
    const progress = analysisProgressMap.get(folders) || 0;
    res.json({ progress });
});

// 기존 분석 시작 엔드포인트 수정
router.post('/analyze', async (req, res) => {
    try {
        console.log('분석 요청 받음:', {
            body: req.body,
            headers: req.headers['content-type']
        });

        // folders 또는 folder 둘 다 처리할 수 있도록
        let folders = req.body.folders || [req.body.folder];
        const io = req.app.get('io');

        if (!folders || !Array.isArray(folders) || folders.length === 0) {
            throw new Error('유효한 폴더 목록이 필요합니다.');
        }

        // 즉시 응답 보내기
        res.json({ status: 'success' });

        const videoDir = path.join(__dirname, '..', 'uploads', 'videos', folders[0]);
        console.log('분석할 비디오 디렉토리:', videoDir);
        
        // 환경변수 설정 (TensorFlow 경고 메시지 숨김)
        const env = {
            ...process.env,
            TF_CPP_MIN_LOG_LEVEL: '2'
        };

        const pythonProcess = spawn('/home/jaeseok/anaconda3/envs/wemeet/bin/python', [
            path.join(__dirname, '..', 'scripts', 'analyze.py'),
            videoDir
        ], {
            env: { ...process.env, TF_CPP_MIN_LOG_LEVEL: '2' }
        });

        // stdout 처리
        pythonProcess.stdout.on('data', (data) => {
            console.log('Python stdout:', data.toString());
            try {
                const lines = data.toString().trim().split('\n');
                for (const line of lines) {
                    try {
                        const parsed = JSON.parse(line);
                        if (parsed.type === 'status') {
                            io.emit('analysisStatus', { message: parsed.message });
                        }
                    } catch (err) {
                        // JSON이 아닌 출력도 로그로 남김
                        console.log('Non-JSON output:', line);
                    }
                }
            } catch (err) {
                console.error('Python 출력 처리 오류:', err);
            }
        });

        // stderr 처리 추가
        pythonProcess.stderr.on('data', (data) => {
            console.error('Python stderr:', data.toString());
        });

        // 프로세스 종료 처리
        pythonProcess.on('close', (code) => {
            console.log('Python 프로세스 종료. 코드:', code);
            if (code === 0) {
                io.emit('analysisStatus', { 
                    message: '✅ 분석이 성공적으로 완료되었습니다.',
                    isComplete: true 
                });
            } else {
                io.emit('analysisError', { 
                    error: `분석 프로세스가 코드 ${code}로 종료되었습니다.` 
                });
            }
        });

    } catch (error) {
        console.error('분석 시작 실패:', error);
        if (!checkResponseSent(res)) {
            res.status(500).json({ error: error.message });
        }
        io.emit('analysisError', { error: error.message });
    }
});

// 분석 결과 API
router.get('/api/analysis-result', async (req, res) => {
    try {
        const folder = req.query.folder;
        console.log('분석 결과 요청 폴더:', folder);
        
        if (folder) {
            const folderPath = path.join(__dirname, '..', 'uploads', 'videos', folder);
            const resultPath = path.join(folderPath, 'analysis_result.json');
            
            if (fs.existsSync(resultPath)) {
                const result = JSON.parse(fs.readFileSync(resultPath, 'utf8'));
                return res.json(result);
            }
        }
        
        const analysisResultsDir = path.join(__dirname, '..', 'data', 'analysis_results');
        const files = fs.readdirSync(analysisResultsDir)
                       .filter(file => file.endsWith('.json'))
                       .sort((a, b) => {
                           return fs.statSync(path.join(analysisResultsDir, b)).mtime.getTime() -
                                  fs.statSync(path.join(analysisResultsDir, a)).mtime.getTime();
                       });
        
        if (files.length > 0) {
            const latestFile = files[0];
            const resultPath = path.join(analysisResultsDir, latestFile);
            const result = JSON.parse(fs.readFileSync(resultPath, 'utf8'));
            return res.json(result);
        }
        
        return res.status(404).json({ error: '분석 결과 파일을 찾을 수 없습니다.' });
        
    } catch (error) {
        console.error('분석 결과 조회 실패:', error);
        res.status(500).json({ error: '분석 결과 조회 실패' });
    }
});

// 설문조사 결과 API
router.get('/api/survey-result', async (req, res) => {
    try {
        const folder = req.query.folder;
        console.log('설문조사 결과 요청 폴더:', folder);
        
        if (!folder) {
            return res.status(400).json({ error: '폴더 파라미터가 필요합니다.' });
        }

        const surveyPath = path.join(__dirname, '..', 'uploads', 'videos', folder, 'survey_result.json');
        console.log('설문조사 파일 경로:', surveyPath);
        
        if (!fs.existsSync(surveyPath)) {
            return res.status(404).json({ error: '설문조사 결과 파일을 찾을 수 없습니다.' });
        }

        const surveyData = await fsPromises.readFile(surveyPath, 'utf8');
        return res.json(JSON.parse(surveyData));

    } catch (error) {
        console.error('설문조사 결과 조회 실패:', error);
        return res.status(500).json({ error: '설문조사 결과 조회 실패' });
    }
});

// 최신 분석 결과 가져오기
router.get('/api/latest-analysis', async (req, res) => {
    try {
        const dataDir = path.join(__dirname, '..', 'data');
        const files = fs.readdirSync(dataDir)
            .filter(file => file.endsWith('.json'))
            .map(file => ({
                name: file,
                time: fs.statSync(path.join(dataDir, file)).mtime.getTime()
            }))
            .sort((a, b) => b.time - a.time);

        if (files.length === 0) {
            return res.status(404).json({ error: '분석 결과 파일이 없습니다.' });
        }

        const latestFile = path.join(dataDir, files[0].name);
        const analysisData = JSON.parse(fs.readFileSync(latestFile, 'utf8'));
        res.json(analysisData);
    } catch (error) {
        console.error('최신 분석 결과 조회 실패:', error);
        res.status(500).json({ error: '분석 결과 조회 실패' });
    }
});

// 설문 결과 가져오기
router.get('/api/survey-result', async (req, res) => {
    try {
        const folder = req.query.folder;
        if (!folder) {
            return res.status(400).json({ error: '폴더 파라미터가 필요합니다.' });
        }

        const surveyPath = path.join(__dirname, '..', 'uploads', 'videos', folder, 'survey_result.json');
        
        if (fs.existsSync(surveyPath)) {
            const surveyData = JSON.parse(fs.readFileSync(surveyPath, 'utf8'));
            res.json(surveyData);
        } else {
            res.status(404).json({ error: '설문 결과 파일이 없습니다.' });
        }
    } catch (error) {
        console.error('설문 결과 조회 실패:', error);
        res.status(500).json({ error: '설문 결과 조회 실패' });
    }
});

// ----------------------------------------------------
module.exports = { router, setIO }; // setIO 함수도 함께 내보내기
