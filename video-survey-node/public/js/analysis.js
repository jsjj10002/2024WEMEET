let eventSource;  // 전역 변수로 선언

document.addEventListener('DOMContentLoaded', async () => {
    const urlParams = new URLSearchParams(window.location.search);
    const folders = urlParams.get('folders').split(',');
    
    // 먼저 썸네일 생성
    try {
        await fetch('/api/create-thumbnails', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ folders })
        });

        // 썸네일 그리드 업데이트
        const thumbnailGrid = document.getElementById('thumbnailGrid');
        folders.forEach(folder => {
            const thumbnailItem = document.createElement('div');
            thumbnailItem.className = 'thumbnail-item';
            thumbnailItem.innerHTML = `
                <img src="/uploads/videos/${folder}/thumbnail.jpg?t=${Date.now()}" 
                     alt="${folder} 썸네일">
                <p>${folder}</p>
            `;
            thumbnailGrid.appendChild(thumbnailItem);
        });

        // 분석 시작
        for (let i = 0; i < folders.length; i++) {
            const folder = folders[i];
            const progress = ((i + 1) / folders.length) * 100;
            
            // 파이썬 스크립트 실행 요청
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ folder })
            });

            if (!response.ok) {
                throw new Error(`분석 실패: ${response.statusText}`);
            }

            const result = await response.json();
            if (result.status === 'error') {
                throw new Error(result.message);
            }

            // 진행률 업데이트
            updateProgress(progress);
        }

    } catch (err) {
        console.error('분석 시작 실패:', err);
        alert('분석을 시작합니다.');
    }
});

function updateProgress(progress) {
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    
    if (progressBar && progressText) {
        progressBar.style.width = `${progress}%`;
        progressText.textContent = `${progress}%`;

        if (progress >= 100) {
            window.location.href = '/api/report';
        }
    }
}

function showResultButton() {
    const analysisResult = document.querySelector('.analysis-result');
    const viewResultBtn = document.getElementById('viewResultBtn');
    
    if (analysisResult && viewResultBtn) {
        analysisResult.style.display = 'block';
        viewResultBtn.addEventListener('click', function() {
            window.location.href = '/api/report';
        });
    }
}

class Analysis {
    constructor() {
        this.progressBar = document.getElementById('analysisProgress');
        this.progressText = document.getElementById('progressText');
        this.resultButton = document.getElementById('viewResultButton');
        
        // Socket.IO 연결
        this.socket = io();
        this.initializeSocketEvents();
        
        // URL에서 폴더 정보 가져오기
        const urlParams = new URLSearchParams(window.location.search);
        this.folders = urlParams.get('folders').split(',');
        
        // 분석 시작
        this.startAnalysis(this.folders);
    }

    initializeSocketEvents() {
        this.socket.on('connect', () => {
            console.log('Socket.IO 연결됨');
        });

        this.socket.on('analysisProgress', (data) => {
            if (data.progress !== undefined) {
                this.updateProgress(data.progress);
            }
        });

        this.socket.on('analysisComplete', (data) => {
            if (this.resultButton) {
                this.resultButton.style.display = 'block';
            }
        });

        this.socket.on('analysisError', (data) => {
            console.log('Analysis status:', data.message || "분석 중...");
        });

        this.socket.on('analysisStatus', (data) => {
            const statusMessage = document.getElementById('status-message');
            statusMessage.textContent = data.message;
            
            if (data.isComplete) {
                document.querySelector('.loading-spinner').style.display = 'none';
            }
        });
    }

    async startAnalysis(folders) {
        try {
            console.log('분석 시작:', folders);
            
            // 썸네일 생성 요청
            const thumbnailResponse = await fetch('/api/create-thumbnails', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ folders })
            });

            if (!thumbnailResponse.ok) {
                throw new Error('썸네일 생성 실패');
            }

            // 썸네일 그리드 업데이트
            this.updateThumbnailGrid(folders);

            // 분석 요청 - 여기서 에러 처리 제거
            await fetch('/analysis/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ folders })
            });

        } catch (error) {
            console.error('분석 시작 실패:', error);
            this.updateStatus(`오류 발생: ${error.message}`, 0);
        }
    }

    updateThumbnailGrid(folders) {
        const thumbnailGrid = document.getElementById('thumbnailGrid');
        if (!thumbnailGrid) return;

        thumbnailGrid.innerHTML = '';
        folders.forEach(folder => {
            const thumbnailItem = document.createElement('div');
            thumbnailItem.className = 'thumbnail-item';
            thumbnailItem.innerHTML = `
                <img src="/uploads/videos/${folder}/thumbnail.jpg?t=${Date.now()}" 
                     alt="${folder} 썸네일">
                <p>${folder}</p>
            `;
            thumbnailGrid.appendChild(thumbnailItem);
        });
    }

    updateProgress(progress) {
        if (this.progressBar) {
            this.progressBar.value = progress;
        }
        if (this.progressText) {
            this.progressText.textContent = `${progress}%`;
        }
        // 100%일 때의 처리
        if (progress === 100) {
            if (this.resultButton) {
                this.resultButton.style.display = 'block';
            }
            // 이벤트 리스너 제거
            this.socket.off('analysisProgress');
            this.socket.off('analysisError');
        }
    }

    updateStatus(message, progress) {
        if (this.progressText) {
            this.progressText.textContent = message;
        }
        if (this.progressBar) {
            this.progressBar.value = progress;
        }
    }

    handleViewResult() {
        const selectedFolders = this.getSelectedFolders();
        console.log('선택된 폴더들:', selectedFolders);

        if (selectedFolders && selectedFolders.length > 0) {
            const folderName = selectedFolders[0];
            const targetUrl = `/analysis-report?folder=${encodeURIComponent(folderName)}`;
            console.log('이동할 URL:', targetUrl);
            window.location.href = targetUrl;
        } else {
            alert('분석할 폴더를 선택해주세요.');
            console.error('선택된 폴더가 없습니다.');
        }
    }

    getSelectedFolders() {
        const checkboxes = document.querySelectorAll('input[name="folders"]:checked');
        const selectedFolders = Array.from(checkboxes).map(cb => cb.value);
        console.log('선택된 체크박스 값들:', selectedFolders);
        return selectedFolders;
    }
}

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    console.log('Analysis 초기화');
    new Analysis();
});

// 다른 이벤트 리스너들 제거

// 분석 결과 보기 버튼 클릭 핸들러
function viewAnalysisResult() {
    const urlParams = new URLSearchParams(window.location.search);
    const originalFolder = urlParams.get('folders'); // 예: AA021
    
    if (originalFolder) {
        window.location.href = `/analysis-report?folder=${originalFolder}`;
    } else {
        console.error('폴더 파라미터가 없습니다.');
    }
}

