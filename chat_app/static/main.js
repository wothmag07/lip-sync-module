document.addEventListener('DOMContentLoaded', () => {
    const micButton = document.querySelector("#mic-button");
    const videoContainer = document.getElementById('video-container');
    const originalIdleVideo = document.getElementById('idle-video');
    let idleVideo = originalIdleVideo;
    const userInput = document.getElementById('user-input');
    const llmOutput = document.getElementById('llm-output');

    let responseVideoPlaying = false;
    let recognition = null;
    let isRecording = false;

    // Try to autoplay the idle video (may fail on some browsers unless muted)
    idleVideo.play().catch(err => console.warn('Idle video autoplay failed:', err));

    // Add a warning if speech recognition isn't supported
    if (!('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)) {
        const warning = document.createElement('div');
        warning.classList.add('warning');
        warning.innerHTML = `<p>Your browser does not support speech recognition. Please use Chrome or Edge for voice input.</p>`;
        document.body.prepend(warning);
        micButton.disabled = true;
        micButton.title = 'Speech recognition not supported in this browser.';
    }

    function setupSpeechRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

        if (!SpeechRecognition) {
            return;
        }

        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            userInput.value = transcript;
            console.log('Recognized speech:', transcript);

            setTimeout(() => {
                if (userInput.value.trim() === transcript) {
                    sendMessage();
                }
            }, 1000);
        };

        recognition.onend = function() {
            isRecording = false;
            micButton.classList.remove('recording');
        };

        recognition.onerror = function(event) {
            console.error('Speech recognition error:', event.error);
            isRecording = false;
            micButton.classList.remove('recording');
        };
    }

    micButton.addEventListener('click', () => {
        if (!recognition) {
            setupSpeechRecognition();
            if (!recognition) return;
        }

        try {
            if (isRecording) {
                recognition.stop();
                isRecording = false;
                micButton.classList.remove('recording');
            } else {
                recognition.start();
                isRecording = true;
                micButton.classList.add('recording');
            }
        } catch (err) {
            console.error('Speech recognition start error:', err);
        }
    });

    userInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            sendMessage();
        }
    });

    async function sendMessage() {
        const userMessage = userInput.value.trim();
        if (!userMessage) return;

        // Display user message
        const userDiv = document.createElement('div');
        userDiv.classList.add('user');
        userDiv.innerHTML = `<p>${userMessage}</p>`;
        llmOutput.appendChild(userDiv);
        userInput.value = '';
        llmOutput.scrollTop = llmOutput.scrollHeight;

        // Typing/loading indicator
        const loadingDiv = document.createElement('div');
        loadingDiv.classList.add('model');
        loadingDiv.setAttribute('id', 'loading');
        loadingDiv.innerHTML = `<p>Typing<span id="dots"></span></p>`;
        llmOutput.appendChild(loadingDiv);
        llmOutput.scrollTop = llmOutput.scrollHeight;

        const stopLoading = startLoadingAnimation();

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: userMessage }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            stopLoading();
            loadingDiv.remove();

            const modelDiv = document.createElement('div');
            modelDiv.classList.add('model');
            modelDiv.innerHTML = `<p>${data.answer}</p>`;
            llmOutput.appendChild(modelDiv);
            llmOutput.scrollTop = llmOutput.scrollHeight;

            // Handle video response
            if (data.videoUrl) {
                idleVideo.pause();
                idleVideo.style.display = 'none';
                responseVideoPlaying = true;

                videoContainer.replaceChildren(); // clear all children

                const videoElement = document.createElement('video');
                videoElement.src = data.videoUrl;
                videoElement.controls = true;
                videoElement.autoplay = true;
                videoElement.style.width = '100%';
                videoElement.style.height = '100%';

                videoElement.addEventListener('ended', () => {
                    responseVideoPlaying = false;
                    videoElement.remove();
                    videoContainer.appendChild(idleVideo);
                    idleVideo.style.display = 'block';
                    idleVideo.currentTime = 0;
                    idleVideo.play().catch(err => console.warn('Idle video autoplay failed:', err));
                });

                videoContainer.appendChild(videoElement);
            }

        } catch (error) {
            console.error('Error:', error);
            stopLoading();
            loadingDiv.remove();

            const errorDiv = document.createElement('div');
            errorDiv.classList.add('model');
            errorDiv.innerHTML = `<p>Sorry, an error occurred: ${error.message}</p>`;
            llmOutput.appendChild(errorDiv);
            llmOutput.scrollTop = llmOutput.scrollHeight;
        }
    }

    function startLoadingAnimation() {
        const loading = document.getElementById('loading');
        const dots = document.getElementById('dots');
        loading.style.display = 'inline-block';

        let dotCount = 0;
        const interval = setInterval(() => {
            dotCount = (dotCount + 1) % 4;
            if (dots) {
                dots.textContent = '.'.repeat(dotCount);
            }
        }, 500);

        return () => {
            clearInterval(interval);
            if (loading) loading.style.display = 'none';
            if (dots) dots.textContent = '';
        };
    }

    window.sendMessage = sendMessage;
});
