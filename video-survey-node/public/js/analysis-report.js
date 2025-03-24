/**
 * analysis-report.js
 * - 화면 로드 시 /analysis/api/emotion-analysis API를 호출하여 최신 감정 분석 결과를 가져옴
 * - 가져온 데이터를 차트에 렌더링
 */

class AnalysisReport {
    constructor() {
        this.data = null;
        this.charts = {};
        this.colors = {
            positive: 'rgba(54, 162, 235, 0.8)',
            negative: 'rgba(255, 99, 132, 0.8)',
            neutral: 'rgba(255, 205, 86, 0.8)',
            positiveLight: 'rgba(54, 162, 235, 0.2)',
            negativeLight: 'rgba(255, 99, 132, 0.2)',
            neutralLight: 'rgba(255, 205, 86, 0.2)'
        };
    }

    async initialize() {
        try {
            await this.loadData();
            this.renderAnalysisSummary();
            this.renderEmotionChart();
            this.renderBehaviorChart();
            this.renderAnalysisTexts();
        } catch (error) {
            console.error('데이터 로드 중 오류 발생:', error);
        }
    }

    async loadData() {
        try {
            const urlParams = new URLSearchParams(window.location.search);
            const folder = urlParams.get('folder');
            console.log('선택된 폴더:', folder);

            // 1. 감정 분석 데이터 로드 (/analysis/api/emotion-analysis)
            const emotionResponse = await fetch('/analysis/api/emotion-analysis');
            if (!emotionResponse.ok) {
                throw new Error('감정 분석 데이터를 불러오는데 실패했습니다.');
            }
            this.data = await emotionResponse.json();
            console.log('감정 분석 데이터:', this.data);

            // 2. 설문 데이터 로드 (/analysis/api/survey-result)
            if (folder) {
                try {
                    // 설문조사 결과 API 호출
                    const surveyResponse = await fetch(`/analysis/api/survey-result?folder=${folder}`);
                    
                    if (surveyResponse.ok) {
                        // 응답이 성공적인 경우
                        const surveyData = await surveyResponse.json();
                        this.data.surveyData = surveyData;
                        console.log('설문조사 데이터 로드 성공:', surveyData);
                        await loadSurveyData(surveyData);
                    } else {
                        // 404 등 에러 응답인 경우
                        const errorData = await surveyResponse.json();
                        console.warn('설문조사 데이터 로드 실패:', errorData.error);
                        document.getElementById('surveySection').style.display = 'none';
                    }
                } catch (error) {
                    // 네트워크 오류 등 예외 발생 시
                    console.error('설문조사 데이터 요청 중 오류:', error);
                    document.getElementById('surveySection').style.display = 'none';
                }
            }

            return this.data;
        } catch (error) {
            console.error('데이터 로드 중 오류 발생:', error);
            throw error;
        }
    }

    renderBehaviorChart() {
        const donutCtx = document.getElementById('behaviorDonutChart');
        const barCtx = document.getElementById('behaviorLineChart');
        if (!donutCtx || !barCtx || !this.data?.summary?.['행동 토탈'] || !this.data?.summary?.['프레임별 행동 분석']) return;

        // 도넛 차트
        const behaviorScores = this.data.summary['행동 토탈'];
        new Chart(donutCtx, {
            type: 'doughnut',
            data: {
                labels: ['긍정적 행동', '부정적 행동'],
                datasets: [{
                    data: [behaviorScores.positive_behavior, behaviorScores.negative_behavior],
                    backgroundColor: [
                        this.colors.positive,
                        this.colors.negative
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });

        // 새로운 가로 막대 차트
        const frameData = this.data.summary['프레임별 행동 분석'];
        
        // 90% 이상인 데이터 카운트
        const highPositiveCount = frameData.filter(frame => frame['Positive (%)'] >= 90).length;
        const highNegativeCount = frameData.filter(frame => frame['Negative (%)'] >= 90).length;

        new Chart(barCtx, {
            type: 'bar',
            data: {
                labels: ['긍정적 제스처', '부정적 제스처'],
                datasets: [{
                    data: [highPositiveCount, highNegativeCount],
                    backgroundColor: [
                        this.colors.positive,
                        this.colors.negative
                    ],
                    borderColor: [
                        this.colors.positive,
                        this.colors.negative
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: '긍정적/부정적 제스처 횟수 (90% 이상)',
                        font: {
                            size: 14
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: '횟수'
                        }
                    },
                    y: {
                        ticks: {
                            font: {
                                size: 14
                            }
                        }
                    }
                }
            }
        });
    }

    renderEmotionChart() {
        const donutCtx = document.getElementById('emotionDonutChart');
        const lineCtx = document.getElementById('emotionLineChart');
        if (!donutCtx || !lineCtx || !this.data?.summary?.['감정 토탈'] || !this.data?.summary?.['그룹별 감정 분석']) return;

        // 도넛 차트
        const emotionScores = this.data.summary['감정 토탈'];
        new Chart(donutCtx, {
            type: 'doughnut',
            data: {
                labels: ['긍정', '부정', '중립'],
                datasets: [{
                    data: [
                        emotionScores.positive,
                        emotionScores.negative,
                        emotionScores.neutral
                    ],
                    backgroundColor: [
                        this.colors.positive,
                        this.colors.negative,
                        this.colors.neutral
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });

        // 라인 차트
        const groupData = this.data.summary['그룹별 감정 분석'];
        new Chart(lineCtx, {
            type: 'line',
            data: {
                labels: groupData.map(group => group.group),
                datasets: [
                    {
                        label: '긍정',
                        data: groupData.map(group => group.positive),
                        borderColor: this.colors.positive,
                        backgroundColor: this.colors.positiveLight,
                        fill: true
                    },
                    {
                        label: '부정',
                        data: groupData.map(group => group.negative),
                        borderColor: this.colors.negative,
                        backgroundColor: this.colors.negativeLight,
                        fill: true
                    },
                    {
                        label: '중립',
                        data: groupData.map(group => group.neutral),
                        borderColor: this.colors.neutral,
                        backgroundColor: this.colors.neutralLight,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    renderAnalysisSummary() {
        if (!this.data?.summary?.['총합 감정+행동']) return;

        const totalScores = this.data.summary['총합 감정+행동'];

        document.getElementById('totalPositive').textContent = `${totalScores.Y_positive.toFixed(1)}%`;
        document.getElementById('totalNegative').textContent = `${totalScores.Y_negative.toFixed(1)}%`;
        document.getElementById('totalNeutral').textContent = `${totalScores.Y_neutral.toFixed(1)}%`;
    }

    renderAnalysisTexts() {
        if (!this.data?.analysis) {
            console.error('분석 데이터가 없습니다:', this.data);
            return;
        }

        // totalAnalysis, emotionAnalysis, behaviorAnalysis ID를 사용하도록 수정
        document.getElementById('totalAnalysis').innerHTML = this.data.analysis.total_analysis;
        document.getElementById('emotionAnalysis').innerHTML = this.data.analysis.emotion_analysis;
        document.getElementById('behaviorAnalysis').innerHTML = this.data.analysis.behavior_analysis;
    }

    createBehaviorCharts(summary) {
        // 행동 토탈 도넛 차트
        const behaviorDonutCtx = document.getElementById('behaviorDonutChart').getContext('2d');
        new Chart(behaviorDonutCtx, {
            type: 'doughnut',
            data: {
                labels: ['긍정적 행동', '부정적 행동'],
                datasets: [{
                    data: [
                        summary['행동 토탈'].positive_behavior,
                        summary['행동 토탈'].negative_behavior
                    ],
                    backgroundColor: [
                        'rgba(54, 162, 235, 0.8)',  // 파란색
                        'rgba(255, 99, 132, 0.8)'   // 빨간색
                    ]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    title: {
                        display: true,
                        text: '전체 행동 분포'
                    }
                }
            }
        });

        // 90% 이상 행동 분석 바 차트
        const behaviorFrames = summary['프레임별 행동 분석'];
        const highPositiveCount = behaviorFrames.filter(frame => frame['Positive (%)'] >= 90).length;
        const highNegativeCount = behaviorFrames.filter(frame => frame['Negative (%)'] >= 90).length;

        const behaviorBarCtx = document.getElementById('behaviorLineChart').getContext('2d');
        new Chart(behaviorBarCtx, {
            type: 'bar',
            data: {
                labels: ['90% 이상 긍정', '90% 이상 부정'],
                datasets: [{
                    data: [highPositiveCount, highNegativeCount],
                    backgroundColor: [
                        'rgba(54, 162, 235, 0.8)',  // 파란색
                        'rgba(255, 99, 132, 0.8)'   // 빨간색
                    ]
                }]
            },
            options: {
                indexAxis: 'y',  // 가로 방향 바 차트
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: '90% 이상 행동 빈도'
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: '발생 횟수'
                        }
                    }
                }
            }
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const report = new AnalysisReport();
    report.initialize();
});

async function loadSurveyData(surveyData) {
    try {
        document.getElementById('surveySection').style.display = 'block';
        
        // 간과 양의 값을 숫자로 변환
        const seasoningMap = {
            '매우 짬': 5,
            '짬': 4,
            '적당함': 3,
            '싱거움': 2,
            '매우 싱거움': 1
        };

        const portionMap = {
            '매우 많음': 5,
            '많음': 4,
            '적당함': 3,
            '적음': 2,
            '매우 적음': 1
        };

        // 색상 설정 함수
        function getBarColor(value, type) {
            if (type === 'normal') {
                switch(value) {
                    case 5: return 'rgba(54, 162, 235, 0.8)';  // 파란색
                    case 4: return 'rgba(75, 192, 192, 0.8)';  // 초록색
                    case 3: return 'rgba(255, 206, 86, 0.8)';  // 노란색
                    case 2: return 'rgba(255, 159, 64, 0.8)';  // 주황색
                    case 1: return 'rgba(255, 99, 132, 0.8)';  // 빨간색
                }
            } else if (type === 'taste') {
                if (value === 3) return 'rgba(54, 162, 235, 0.8)';  // 파란색 (적당함)
                if (value === 2 || value === 4) return 'rgba(75, 192, 192, 0.8)';  // 초록색
                return 'rgba(255, 206, 86, 0.8)';  // 노란색
            }
        }

        const ctx = document.getElementById('satisfactionChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['맛', '서비스', '청결도', '간', '양'],
                datasets: [{
                    data: [
                        surveyData.taste,
                        surveyData.service,
                        surveyData.cleanliness,
                        seasoningMap[surveyData.seasoning],
                        portionMap[surveyData.portion]
                    ],
                    backgroundColor: [
                        getBarColor(surveyData.taste, 'normal'),
                        getBarColor(surveyData.service, 'normal'),
                        getBarColor(surveyData.cleanliness, 'normal'),
                        getBarColor(seasoningMap[surveyData.seasoning], 'taste'),
                        getBarColor(portionMap[surveyData.portion], 'taste')
                    ],
                    barThickness: 20
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const value = context.raw;
                                const index = context.dataIndex;
                                
                                if (index === 3) { // 간
                                    const seasoningLabels = ['매우 싱거움', '싱거움', '적당함', '짬', '매우 짬'];
                                    return seasoningLabels[Math.round(value) - 1];
                                } else if (index === 4) { // 양
                                    const portionLabels = ['매우 적음', '적음', '적당함', '많음', '매우 많음'];
                                    return portionLabels[Math.round(value) - 1];
                                }
                                return value;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 5,
                        ticks: {
                            stepSize: 1
                        }
                    },
                    y: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });

        // 피드백 텍스트 표시
        const feedbackText = document.getElementById('feedbackText');
        if (feedbackText && surveyData.feedback) {
            feedbackText.textContent = surveyData.feedback;
            console.log('피드백 내용:', surveyData.feedback); // 디버깅용
        }

    } catch (error) {
        console.error('설문 차트 생성 실패:', error);
    }
}