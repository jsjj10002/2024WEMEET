const express = require('express');
const router = express.Router();
const Survey = require('../models/Survey');
const fs = require('fs').promises;
const path = require('path');

// 숫자를 문자열로 변환하는 함수들
const convertSeasoningToString = (value) => {
    const map = {
        1: '매우 싱거움',
        2: '싱거움',
        3: '적당함',
        4: '짬',
        5: '매우 짬'
    };
    return map[value] || '알 수 없음';
};

const convertPortionToString = (value) => {
    const map = {
        1: '매우 적음',
        2: '적음',
        3: '적당함',
        4: '많음',
        5: '매우 많음'
    };
    return map[value] || '알 수 없음';
};

router.post('/submit-survey', async (req, res) => {
    try {
        console.log('받은 설문 데이터:', req.body);

        // 설문 데이터 준비
        const surveyData = {
            customerId: req.body.customerId,
            taste: parseInt(req.body.taste),  // 숫자 유지
            seasoning: convertSeasoningToString(parseInt(req.body.seasoning)),  // 문자열로 변환
            service: parseInt(req.body.service),  // 숫자 유지
            cleanliness: parseInt(req.body.cleanliness),  // 숫자 유지
            portion: convertPortionToString(parseInt(req.body.portion)),  // 문자열로 변환
            feedback: req.body.feedback
        };

        // MongoDB에 저장
        const survey = new Survey(surveyData);
        await survey.save();

        // 파일 저장 경로 설정
        const customerDir = path.join('uploads/videos', surveyData.customerId);

        // CSV 파일 생성
        const csvContent = `고객ID,맛,간,서비스,청결도,양,피드백\n${surveyData.customerId},${surveyData.taste},${surveyData.seasoning},${surveyData.service},${surveyData.cleanliness},${surveyData.portion},"${surveyData.feedback}"`;
        await fs.writeFile(path.join(customerDir, `${surveyData.customerId}.csv`), csvContent);

        // JSON 파일 생성
        await fs.writeFile(
            path.join(customerDir, `${surveyData.customerId}.json`),
            JSON.stringify(surveyData, null, 2)
        );

        res.json({ success: true, message: '설문이 성공적으로 저장되었습니다.' });
    } catch (err) {
        console.error('설문 저장 오류:', err);
        res.status(500).json({ success: false, error: err.message });
    }
});

module.exports = router;