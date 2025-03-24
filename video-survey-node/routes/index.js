const express = require('express');
const router = express.Router();
const path = require('path');
const fs = require('fs').promises;

// 홈 페이지
router.get('/', (req, res) => {
    res.render('home');
});

// 녹화 페이지
router.get('/recording', (req, res) => {
    res.render('recording');
});

// 매니저 로그인 페이지
router.get('/manager-login', (req, res) => {
    res.render('manager-login');
});

// 매니저 로그인 처리
router.post('/api/manager-login', (req, res) => {
    const { code } = req.body;
    
    if (code === '0000') {
        res.json({ success: true });
    } else {
        res.json({ success: false });
    }
});

// 통계 페이지 라우트 추가
router.get('/statistics', (req, res) => {
    res.render('statistics');
});

// 매니저 대시보드 페이지
router.get('/manager-dashboard', (req, res) => {
    res.render('manager-dashboard');  
});

// 분석 페이지 라우트
router.get('/analysis', async (req, res) => {
    try {
        // 영상 업로드 기본 경로
        const videosPath = path.join(__dirname, '..', 'uploads', 'videos');
        
        // 폴더 목록 읽기
        // (uploads/videos 폴더가 없을 경우 에러 처리 필요)
        const folders = await fs.readdir(videosPath);

        // ejs 템플릿에 folders 배열 전달
        res.render('analysis', { folders });
    } catch (error) {
        console.error('/analysis 라우트 에러:', error);
        
        // 폴더 목록을 불러오는 데 실패했을 경우에도 ejs 생성은 진행 (빈 배열 전달)
        res.render('analysis', { folders: [] });
    }
});

module.exports = router;