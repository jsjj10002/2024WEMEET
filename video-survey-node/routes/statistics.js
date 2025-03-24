const express = require('express');
const router = express.Router();
const Survey = require('../models/Survey');

// 전체 통계 데이터 가져오기
router.get('/survey-stats', async (req, res) => {
    try {
        // 모든 설문 데이터 가져오기
        const surveys = await Survey.find();
        
        // 총 응답 수
        const totalResponses = surveys.length;

        // 맛 평가 통계
        const tasteStats = {
            1: 0, 2: 0, 3: 0, 4: 0, 5: 0
        };
        surveys.forEach(survey => {
            tasteStats[survey.taste]++;
        });

        // 간 평가 통계
        const seasoningStats = {};
        surveys.forEach(survey => {
            seasoningStats[survey.seasoning] = (seasoningStats[survey.seasoning] || 0) + 1;
        });

        // 서비스 평가 통계
        const serviceStats = {
            1: 0, 2: 0, 3: 0, 4: 0, 5: 0
        };
        surveys.forEach(survey => {
            serviceStats[survey.service]++;
        });

        // 청결도 통계
        const cleanlinessStats = {
            1: 0, 2: 0, 3: 0, 4: 0, 5: 0
        };
        surveys.forEach(survey => {
            cleanlinessStats[survey.cleanliness]++;
        });

        // 양 평가 통계
        const portionStats = {};
        surveys.forEach(survey => {
            portionStats[survey.portion] = (portionStats[survey.portion] || 0) + 1;
        });

        // 전체 평균 만족도 계산 (맛, 서비스, 청결도의 평균)
        const avgSatisfaction = surveys.reduce((acc, survey) => {
            return acc + (survey.taste + survey.service + survey.cleanliness) / 3;
        }, 0) / totalResponses;

        // 최근 피드백 목록 (최대 10개)
        const recentFeedbacks = surveys
            .filter(survey => survey.feedback)
            .map(survey => ({
                customerId: survey.customerId,
                feedback: survey.feedback
            }))
            .slice(-10);

        res.json({
            totalResponses,
            avgSatisfaction: avgSatisfaction.toFixed(1),
            tasteStats,
            seasoningStats,
            serviceStats,
            cleanlinessStats,
            portionStats,
            recentFeedbacks
        });
    } catch (err) {
        console.error('통계 데이터 조회 오류:', err);
        res.status(500).json({ error: '통계 데이터를 가져오는데 실패했습니다.' });
    }
});

module.exports = router;