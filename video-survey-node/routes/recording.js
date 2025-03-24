const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const Customer = require('../models/Customer');
const idGenerator = require('../utils/idGenerator');

// 비디오 저장 설정
const storage = multer.diskStorage({
    destination: async function (req, file, cb) {
        try {
            // customerId가 이미 있는 경우 그대로 사용
            let customerId = req.body.customerId;
            
            // customerId가 없는 경우에만 새로 생성
            if (!customerId) {
                const lastCustomer = await Customer.findOne().sort({ customerId: -1 });
                customerId = idGenerator.generateNextId(lastCustomer ? lastCustomer.customerId : null);
                
                const customer = new Customer({ customerId: customerId });
                await customer.save();
                
                console.log('새로 생성된 고객 ID:', customerId);
            }

            // 고객별 디렉토리 생성
            const customerDir = path.join('uploads/videos', customerId);
            if (!fs.existsSync(customerDir)) {
                fs.mkdirSync(customerDir, { recursive: true });
            }

            req.customerId = customerId;
            cb(null, customerDir);
        } catch (err) {
            cb(err);
        }
    },
    filename: function (req, file, cb) {
        const now = new Date();
        const dateStr = now.toISOString().slice(0,10).replace(/-/g,'');
        const timeStr = now.toTimeString().slice(0,5).replace(':', '');
        const fileName = `${dateStr}_${timeStr}.mp4`;
        cb(null, fileName);
    }
});

const upload = multer({ storage: storage });

// 비디오 청크 업로드 라우트
router.post('/upload-chunk', upload.single('video'), async (req, res) => {
    try {
        if (!req.file) {
            throw new Error('파일이 없습니다.');
        }
        
        console.log('업로드 요청 수신');
        console.log('사용된 고객 ID:', req.customerId);

        res.json({ 
            success: true, 
            customerId: req.customerId,
            filename: req.file.filename 
        });
    } catch (err) {
        console.error('업로드 오류:', err);
        res.status(400).json({ success: false, error: err.message });
    }
});

module.exports = router;