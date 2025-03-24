let mediaRecorder;
let recordedChunks = [];

document.addEventListener('DOMContentLoaded', async () => {
    const videoPreview = document.getElementById('videoPreview');
    const videoPlaceholder = document.getElementById('videoPlaceholder');
    const startButton = document.getElementById('startRecording');
    const stopButton = document.getElementById('stopRecording');

    startButton.addEventListener('click', async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: true, 
                audio: true 
            });
            videoPreview.srcObject = stream;
            
            videoPreview.style.display = 'block';
            videoPlaceholder.style.display = 'none';

            mediaRecorder = new MediaRecorder(stream);
            recordedChunks = [];
            
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    recordedChunks.push(event.data);
                    uploadChunk();
                }
            };

            mediaRecorder.start(180000);
            startButton.disabled = true;
            stopButton.disabled = false;

        } catch (err) {
            console.error('카메라 접근 오류:', err);
            alert('카메라 접근 권한이 필요합니다.');
        }
    });

    stopButton.addEventListener('click', async () => {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            const tracks = videoPreview.srcObject.getTracks();
            tracks.forEach(track => track.stop());
        }
        startButton.disabled = false;
        stopButton.disabled = true;

        await new Promise(resolve => setTimeout(resolve, 1000));

        const savedCustomerId = localStorage.getItem('customerId');
        console.log('녹화 종료 시 저장된 고객 ID:', savedCustomerId);

        if (!savedCustomerId) {
            alert('고객 ID가 저장되지 않았습니다. 다시 시도해주세요.');
            window.location.href = '/';
            return;
        }

        window.location.href = '/survey';
    });
});

async function uploadChunk() {
    try {
        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        const formData = new FormData();
        formData.append('video', blob);

        const response = await fetch('/api/upload-chunk', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('업로드 실패');
        }

        const data = await response.json();
        console.log('서버 응답:', data);

        if (data.customerId) {
            localStorage.setItem('customerId', data.customerId);
            console.log('청크 업로드 후 저장된 고객 ID:', data.customerId);
        } else {
            console.error('서버에서 customerId를 받지 못했습니다.');
        }
        
        recordedChunks = [];

    } catch (err) {
        console.error('업로드 오류:', err);
    }
}