document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.querySelector('.chat-window');
    const chatButton = document.querySelector('.chat-button');
    const videoContainer = document.getElementById('video-container');
    // Store the original idle video element for reference
    const originalIdleVideo = document.getElementById('idle-video');
    // We need to keep the idle video separate from the response video
    let idleVideo = originalIdleVideo;
    const userInput = document.getElementById('user-input');
    const llmOutput = document.getElementById('llm-output');
    
    // State management
    let responseVideoPlaying = false;

    // Start playing the idle video immediately when the page loads
    idleVideo.play();

    function toggleChat() {
        chatWindow.classList.toggle('hidden');
        chatButton.classList.toggle('hidden');
    }

    // Speech recognition setup
    let recognition = null;
    let isRecording = false;
    
    // Initialize speech recognition if browser supports it
    function setupSpeechRecognition() {
        if ('webkitSpeechRecognition' in window) {
            recognition = new webkitSpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';
            
            recognition.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                userInput.value = transcript;
                console.log('Recognized speech:', transcript);
                
                // Auto-send the message after a short delay to allow the user to see what was transcribed
                setTimeout(() => {
                    if (userInput.value.trim() === transcript) {  // Only if the input wasn't changed manually
                        sendMessage();
                    }
                }, 1000);
            };
            
            recognition.onend = function() {
                isRecording = false;
                document.querySelector('#mic-button').classList.remove('recording');
            };
            
            recognition.onerror = function(event) {
                console.error('Speech recognition error:', event.error);
                isRecording = false;
                document.querySelector('#mic-button').classList.remove('recording');
            };
        } else {
            console.error('Speech recognition not supported in this browser');
        }
    }
    
    // Toggle microphone recording
    function toggleMicrophone() {
        if (!recognition) {
            setupSpeechRecognition();
            if (!recognition) return; // Exit if speech recognition couldn't be initialized
        }
        
        if (isRecording) {
            recognition.stop();
            isRecording = false;
            document.querySelector('#mic-button').classList.remove('recording');
        } else {
            recognition.start();
            isRecording = true;
            document.querySelector('#mic-button').classList.add('recording');
        }
    }

    // Add event listener for Enter key on input field
    userInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            sendMessage();
        }
    });

    async function sendMessage() {
        const userMessage = userInput.value.trim();
        if (!userMessage) return;
        
        // Append user's message to chat
        const userDiv = document.createElement('div');
        userDiv.classList.add('user');
        userDiv.innerHTML = `<p>${userMessage}</p>`;
        llmOutput.appendChild(userDiv);

        // Clear input field
        userInput.value = '';
        
        // Auto-scroll chat to bottom
        llmOutput.scrollTop = llmOutput.scrollHeight;

        // Keep the idle video playing while waiting for the response
        // (idle video should already be playing)

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: userMessage }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            let data;
            try {
                data = await response.json();
            } catch (jsonError) {
                throw new Error('Invalid JSON response from server');
            }

            // Append model response to chat
            const modelDiv = document.createElement('div');
            modelDiv.classList.add('model');
            modelDiv.innerHTML = `<p>${data.answer}</p>`;
            llmOutput.appendChild(modelDiv);
            
            // Auto-scroll chat to bottom
            llmOutput.scrollTop = llmOutput.scrollHeight;

            // Display the response video if available
            if (data.videoUrl) {
                // Hide the idle video while response video plays
                idleVideo.style.display = 'none';
                idleVideo.pause();
                
                // Set flag that response video is playing
                responseVideoPlaying = true;
                
                // Clear any previous response videos (but don't use innerHTML='')
                // Instead, preserve the idle video and remove all other children
                while (videoContainer.firstChild) {
                    videoContainer.removeChild(videoContainer.firstChild);
                }
                
                // Add the idle video back to the container
                videoContainer.appendChild(idleVideo);
                
                // Create new video element for the response
                const videoElement = document.createElement('video');
                videoElement.src = data.videoUrl;
                videoElement.controls = true;
                videoElement.autoplay = true;
                videoElement.style.width = '100%';
                videoElement.style.height = '100%';
                
                // Add event listener for when response video ends
                videoElement.addEventListener('ended', () => {
                    // Video finished playing, return to idle state
                    responseVideoPlaying = false;
                    
                    // Remove the response video
                    videoElement.remove();
                    
                    // Show and play the idle video again
                    idleVideo.style.display = 'block';
                    idleVideo.currentTime = 0; // Start from beginning
                    idleVideo.play();
                });
                
                videoContainer.appendChild(videoElement);
            }
        } catch (error) {
            console.error('Error:', error);
            const errorDiv = document.createElement('div');
            errorDiv.classList.add('model');
            errorDiv.innerHTML = `<p>Sorry, an error occurred: ${error.message}</p>`;
            llmOutput.appendChild(errorDiv);
            
            // Auto-scroll chat to bottom
            llmOutput.scrollTop = llmOutput.scrollHeight;
        }
    }

    window.toggleChat = toggleChat;
    window.toggleMicrophone = toggleMicrophone;
    window.sendMessage = sendMessage;
    
    // Initialize speech recognition when the page loads
    setupSpeechRecognition();
});
