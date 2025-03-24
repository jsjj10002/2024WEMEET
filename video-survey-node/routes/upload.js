const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fs = require('fs');

// uploads/videos 디렉토리가 없으면 생성
const uploadDir = path.join(__dirname, '../uploads/videos');
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}

// Multer 설정
const storage = multer.diskStorage({
    destination: function(req, file, cb) {
        // URL 파라미터에서 folderName 가져오기
        const folderName = req.params.folderName;
        console.log('폴더명:', folderName); // 디버깅용 로그
        
        if (!folderName) {
            return cb(new Error('폴더명이 지정되지 않았습니다.'));
        }
        
        const uploadPath = path.join(uploadDir, folderName);
        console.log('업로드 경로:', uploadPath); // 디버깅용 로그
        
        // 해당 폴더가 없으면 생성
        if (!fs.existsSync(uploadPath)) {
            fs.mkdirSync(uploadPath, { recursive: true });
        }
        
        cb(null, uploadPath);
    },
    filename: function(req, file, cb) {
        // 원본 파일명 유지
        cb(null, file.originalname);
    }
});

const upload = multer({ 
    storage: storage,
    fileFilter: function(req, file, cb) {
        if (file.mimetype.startsWith('video/mp4')) {
            cb(null, true);
        } else {
            cb(new Error('MP4 파일만 업로드 가능합니다.'));
        }
    }
});

// 비디오 업로드 라우트 - URL 파라미터로 folderName 받기
router.post('/upload-videos/:folderName', upload.array('videos'), (req, res) => {
    try {
        console.log('파일 업로드 요청 받음:', req.params.folderName); // 디버깅용 로그
        
        if (!req.files || req.files.length === 0) {
            throw new Error('업로드된 파일이 없습니다.');
        }

        res.json({
            success: true,
            message: '업로드 성공',
            folderName: req.params.folderName,
            files: req.files.map(f => f.filename)
        });
    } catch (error) {
        console.error('Upload error:', error);
        res.status(400).json({
            success: false,
            error: error.message
        });
    }
});

module.exports = router;
