document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.getElementById('loginForm');

    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const code = document.getElementById('code').value;

        try {
            const response = await fetch('/api/manager-login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ code })
            });

            if (!response.ok) {
                throw new Error('서버 오류가 발생했습니다.');
            }

            const data = await response.json();

            if (data.success) {
                window.location.href = '/manager-dashboard';
            } else {
                alert('입력하신 코드가 틀렸습니다.');
                window.location.href = '/';
            }
        } catch (err) {
            console.error('로그인 오류:', err);
            alert('로그인 중 오류가 발생했습니다.');
        }
    });
});