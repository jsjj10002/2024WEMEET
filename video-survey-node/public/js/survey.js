document.addEventListener('DOMContentLoaded', () => {
    const surveyForm = document.getElementById('surveyForm');
    
    console.log('현재 저장된 고객 ID:', localStorage.getItem('customerId'));

    surveyForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData(surveyForm);
        const customerId = localStorage.getItem('customerId');

        console.log('설문 제출 시 고객 ID:', customerId);

        if (!customerId) {
            alert('고객 ID를 찾을 수 없습니다.');
            return;
        }

        const surveyData = {
            taste: parseInt(formData.get('taste')),
            seasoning: parseInt(formData.get('seasoning')),
            service: parseInt(formData.get('service')),
            cleanliness: parseInt(formData.get('cleanliness')),
            portion: parseInt(formData.get('portion')),
            feedback: formData.get('feedback'),
            customerId: customerId
        };

        try {
            const response = await fetch('/api/submit-survey', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(surveyData)
            });

            if (!response.ok) {
                throw new Error('설문 제출 실패');
            }

            alert('설문이 성공적으로 제출되었습니다.');
            window.location.href = '/';
        } catch (err) {
            console.error('설문 제출 오류:', err);
            alert('설문 제출 중 오류가 발생했습니다.');
        }
    });
});