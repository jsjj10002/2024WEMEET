const express = require('express');
const router = express.Router();
const path = require('path');
const fs = require('fs').promises;

// 폴더 목록 가져오기
router.get('/get-folders', async (req, res) => {
    try {
        const videosPath = path.join(__dirname, '..', 'uploads', 'videos');
        const folders = await fs.readdir(videosPath);
        res.json(folders);
    } catch (error) {
        console.error('폴더 목록 조회 실패:', error);
        res.status(500).json({ error: '폴더 목록을 가져오는데 실패했습니다.' });
    }
});

// 다른 경로 선택 시 보안 메시지 반환
router.post('/select-path', (req, res) => {
    res.status(403).json({ 
        error: '보안상의 이유로 기본 경로만 접근 가능합니다.' 
    });
});

module.exports = router;