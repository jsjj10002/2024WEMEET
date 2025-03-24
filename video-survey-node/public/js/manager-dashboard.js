document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById('folderModal');
    const selectFolderBtn = document.getElementById('selectFolderBtn');
    const completeFolderSelection = document.getElementById('completeFolderSelection');
    const upload = document.getElementById('upload');
    const folderList = document.getElementById('folderList');
    const selectedCount = document.getElementById('selectedCount');
    const startAnalysisBtn = document.getElementById('startAnalysisBtn');
    const exitBtn = document.querySelector('.exit-btn');
    
    let selectedFolders = new Set();

    // 폴더 선택 버튼 클릭
    selectFolderBtn.addEventListener('click', async () => {
        showLoading();
        modal.style.display = 'block';
        await loadFolders();
        hideLoading();
    });

    // 나가기 버튼 클릭
    exitBtn.addEventListener('click', () => {
        window.location.href = '/';
    });

    // 모달 외부 클릭 시 닫기
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });

    // 업로드 버튼 클릭 
    upload.addEventListener('click', async () => {
        // file input 엘리먼트 생성
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.multiple = true;
        fileInput.accept = '.mp4';
        
        fileInput.addEventListener('change', async (e) => {
            if (e.target.files.length === 0) return;
            
            showLoading();
            
            try {
                // 현재 시각으로 폴더명 생성 (YYMMDD_HHMM 형식)
                const now = new Date();
                const folderName = now.getFullYear().toString().slice(-2) +
                    String(now.getMonth() + 1).padStart(2, '0') +
                    String(now.getDate()).padStart(2, '0') + '_' +
                    String(now.getHours()).padStart(2, '0') +
                    String(now.getMinutes()).padStart(2, '0') +
                    String(now.getSeconds()).padStart(2, '0') +
                    String(now.getMilliseconds()).padStart(3, '0'); 
        
                const formData = new FormData();
                for (let file of e.target.files) {
                    formData.append('videos', file);
                }

                console.log('업로드 시작:', folderName); // 디버깅용 로그

                const response = await fetch(`/upload/upload-videos/${folderName}`, {
                    method: 'POST',
                    body: formData
                });

                console.log('서버 응답:', response.status); // 디버깅용 로그

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || '업로드 실패');
                }

                const data = await response.json();
                console.log('업로드 성공:', data); // 디버깅용 로그

                // 성공 시 폴더 목록 새로고침
                await loadFolders();
                
                // 새로 생성된 폴더 자동 선택
                selectedFolders.clear();
                selectedFolders.add(folderName);
                
                // 선택된 폴더 수 업데이트
                const count = selectedFolders.size;
                selectedCount.textContent = `${count}명의 영상이 선택되었습니다.`;
                startAnalysisBtn.style.display = 'block';
                
                alert('업로드가 완료되었습니다.');
                modal.style.display = 'none';
    
            } catch (err) {
                console.error('Upload error:', err);
                alert(err.message || '업로드 중 오류가 발생했습니다.');
            } finally {
                hideLoading();
            }
        });
    
        fileInput.click();
    });
    

    // 폴더 목록 로드
    async function loadFolders() {
        try {
            const response = await fetch('/api/get-folders');
            if (!response.ok) {
                throw new Error('폴더 목록을 가져오는데 실패했습니다.');
            }

            const folders = await response.json();
            
            folderList.innerHTML = '';
            if (folders.length === 0) {
                folderList.innerHTML = '<div class="no-folders">저장된 영상이 없습니다.</div>';
                return;
            }

            folders.forEach(folder => {
                const folderElement = createFolderElement(folder);
                folderList.appendChild(folderElement);
            });
        } catch (err) {
            console.error('폴더 목록 로드 실패:', err);
            alert(err.message);
        }
    }

    // 로딩 표시
    function showLoading() {
        const loading = document.createElement('div');
        loading.className = 'loading';
        loading.id = 'loadingIndicator';
        loading.textContent = '로딩 중...';
        document.body.appendChild(loading);
        loading.style.display = 'block';
    }

    function hideLoading() {
        const loading = document.getElementById('loadingIndicator');
        if (loading) {
            loading.remove();
        }
    }

    // 폴더 요소 생성
    function createFolderElement(folderName) {
        const div = document.createElement('div');
        div.className = 'folder-item';
        div.textContent = folderName;
        
        if (selectedFolders.has(folderName)) {
            div.classList.add('selected');
        }

        div.addEventListener('click', () => {
            div.classList.toggle('selected');
            if (selectedFolders.has(folderName)) {
                selectedFolders.delete(folderName);
            } else {
                selectedFolders.add(folderName);
            }
        });

        return div;
    }

    // 선택 완료 버튼 클릭
    completeFolderSelection.addEventListener('click', () => {
        modal.style.display = 'none';
        const count = selectedFolders.size;
        if (count > 0) {
            selectedCount.textContent = `${count}명의 영상이 선택되었습니다.`;
            startAnalysisBtn.style.display = 'block';
        } else {
            selectedCount.textContent = '';
            startAnalysisBtn.style.display = 'none';
        }
    });

    // 분석 시작 버튼 클릭
    startAnalysisBtn.addEventListener('click', async () => {
        if (selectedFolders.size === 0) {
            alert('분석할 폴더를 선택해주세요.');
            return;
        }
        
        const foldersParam = Array.from(selectedFolders).join(',');
        
        try {
            // 분석 시작 요청
            const response = await fetch('/analysis/start-analysis', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ folders: Array.from(selectedFolders) })
            });

            if (!response.ok) {
                throw new Error('분석 시작 실패');
            }

            // analysis 페이지로 이동 (analysis-report가 아님)
            window.location.href = `/analysis?folders=${foldersParam}`;
            
        } catch (error) {
            console.error('분석 시작 오류:', error);
            alert('분석 시작 중 오류가 발생했습니다.');
        }
    });
});
