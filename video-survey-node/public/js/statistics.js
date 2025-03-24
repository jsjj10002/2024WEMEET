document.addEventListener('DOMContentLoaded', async () => {
    try {
        // 통계 데이터 가져오기
        const response = await fetch('/api/survey-stats');
        const stats = await response.json();

        // 요약 통계 업데이트
        document.getElementById('totalResponses').textContent = stats.totalResponses;
        document.getElementById('avgSatisfaction').textContent = `${stats.avgSatisfaction} / 5.0`;

        // 차트 색상 설정
        const chartColors = {
            background: [
                'rgba(255, 99, 132, 0.2)',
                'rgba(54, 162, 235, 0.2)',
                'rgba(255, 206, 86, 0.2)',
                'rgba(75, 192, 192, 0.2)',
                'rgba(153, 102, 255, 0.2)'
            ],
            border: [
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(153, 102, 255, 1)'
            ]
        };

        // 맛 평가 차트
        new Chart(document.getElementById('tasteChart'), {
            type: 'bar',
            data: {
                labels: ['매우 나쁨', '나쁨', '보통', '좋음', '매우 좋음'],
                datasets: [{
                    label: '응답 수',
                    data: Object.values(stats.tasteStats),
                    backgroundColor: chartColors.background,
                    borderColor: chartColors.border,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                }
            }
        });

        // 간 평가 차트
        new Chart(document.getElementById('seasoningChart'), {
            type: 'pie',
            data: {
                labels: Object.keys(stats.seasoningStats),
                datasets: [{
                    data: Object.values(stats.seasoningStats),
                    backgroundColor: chartColors.background,
                    borderColor: chartColors.border,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true
            }
        });

        // 서비스 평가 차트
        new Chart(document.getElementById('serviceChart'), {
            type: 'bar',
            data: {
                labels: ['매우 나쁨', '나쁨', '보통', '좋음', '매우 좋음'],
                datasets: [{
                    label: '응답 수',
                    data: Object.values(stats.serviceStats),
                    backgroundColor: chartColors.background,
                    borderColor: chartColors.border,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                }
            }
        });

        // 청결도 차트
        new Chart(document.getElementById('cleanlinessChart'), {
            type: 'bar',
            data: {
                labels: ['매우 나쁨', '나쁨', '보통', '좋음', '매우 좋음'],
                datasets: [{
                    label: '응답 수',
                    data: Object.values(stats.cleanlinessStats),
                    backgroundColor: chartColors.background,
                    borderColor: chartColors.border,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                }
            }
        });

        // 양 만족도 차트
        new Chart(document.getElementById('portionChart'), {
            type: 'pie',
            data: {
                labels: Object.keys(stats.portionStats),
                datasets: [{
                    data: Object.values(stats.portionStats),
                    backgroundColor: chartColors.background,
                    borderColor: chartColors.border,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true
            }
        });

        // 피드백 목록 업데이트
        const feedbackList = document.getElementById('feedbackList');
        stats.recentFeedbacks.forEach(feedback => {
            const feedbackItem = document.createElement('div');
            feedbackItem.className = 'feedback-item';
            feedbackItem.innerHTML = `
                <strong>${feedback.customerId}</strong>: ${feedback.feedback}
            `;
            feedbackList.appendChild(feedbackItem);
        });

    } catch (err) {
        console.error('통계 데이터 로드 실패:', err);
        alert('통계 데이터를 불러오는데 실패했습니다.');
    }
});