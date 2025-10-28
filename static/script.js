// Global variables
let currentAudioUrl = null;

// DOM Elements
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const chatMessages = document.getElementById('chatMessages');
const loading = document.getElementById('loading');
const audioModal = document.getElementById('audioModal');
const audioPlayer = document.getElementById('audioPlayer');
const downloadBtn = document.getElementById('downloadBtn');

// Event listeners
messageInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Send message function
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    const sessionId = document.getElementById('sessionId').value;
    const conversationId = document.getElementById('conversationId').value;
    const token = document.getElementById('token').value;

    // Add user message to chat
    addMessage(message, 'user');
    
    // Clear input and disable send button
    messageInput.value = '';
    sendButton.disabled = true;
    showLoading(true);

    try {
        // Prepare request data
        const requestData = {
            conversationId: conversationId,
            sessionId: sessionId,
            token: token,
            message: message
        };

        // Send request to API
        const response = await fetch('/ai/chat/text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        // Add bot response to chat
        addMessage(data.response, 'bot', data.audio_file_url);

        // Show success notification
        showNotification('Tin nhắn đã được gửi thành công!', 'success');

    } catch (error) {
        console.error('Error sending message:', error);
        addMessage('Xin lỗi, đã có lỗi xảy ra khi xử lý tin nhắn của bạn. Vui lòng thử lại.', 'bot');
        showNotification('Lỗi khi gửi tin nhắn: ' + error.message, 'error');
    } finally {
        sendButton.disabled = false;
        showLoading(false);
    }
}

// Add message to chat
function addMessage(text, sender, audioUrl = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const icon = sender === 'user' ? 'fas fa-user' : 'fas fa-robot';
    
    let audioControls = '';
    if (audioUrl && sender === 'bot') {
        audioControls = `
            <div class="audio-controls-inline">
                <button class="audio-btn" onclick="playInlineAudio('${audioUrl}')">
                    <i class="fas fa-play"></i> Phát Audio
                </button>
                <button class="audio-btn" onclick="openAudioModal('${audioUrl}')">
                    <i class="fas fa-volume-up"></i> Xem Audio
                </button>
                <button class="audio-btn" onclick="downloadAudio('${audioUrl}')">
                    <i class="fas fa-download"></i> Tải về
                </button>
            </div>
        `;
    }
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <i class="${icon}"></i>
            <div class="text">
                ${text}
                ${audioControls}
            </div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Show/hide loading
function showLoading(show) {
    loading.style.display = show ? 'block' : 'none';
}

// Play audio inline
function playInlineAudio(audioUrl) {
    const audio = new Audio(audioUrl);
    audio.play().catch(error => {
        console.error('Error playing audio:', error);
        showNotification('Không thể phát audio', 'error');
    });
}

// Open audio modal
function openAudioModal(audioUrl) {
    currentAudioUrl = audioUrl;
    audioPlayer.src = audioUrl;
    audioModal.style.display = 'flex';
    
    // Set download button
    downloadBtn.onclick = () => downloadAudio(audioUrl);
}

// Close audio modal
function closeAudioModal() {
    audioModal.style.display = 'none';
    audioPlayer.pause();
    currentAudioUrl = null;
}

// Play audio in modal
function playAudio() {
    if (audioPlayer.src) {
        audioPlayer.play().catch(error => {
            console.error('Error playing audio:', error);
            showNotification('Không thể phát audio', 'error');
        });
    }
}

// Download audio file
function downloadAudio(audioUrl) {
    try {
        // Create a temporary anchor element
        const link = document.createElement('a');
        link.href = audioUrl;
        
        // Extract filename from URL or create a default name
        const urlParts = audioUrl.split('/');
        const filename = urlParts[urlParts.length - 1] || `audio_${new Date().getTime()}.wav`;
        
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        showNotification('Đang tải file audio...', 'success');
    } catch (error) {
        console.error('Error downloading audio:', error);
        showNotification('Lỗi khi tải audio: ' + error.message, 'error');
    }
}

// Show notification
function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existingNotifications = document.querySelectorAll('.notification');
    existingNotifications.forEach(notification => notification.remove());
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas ${getNotificationIcon(type)}"></i>
            <span>${message}</span>
            <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    // Add notification styles
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${getNotificationColor(type)};
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 1001;
        animation: slideInRight 0.3s ease;
        max-width: 400px;
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

function getNotificationIcon(type) {
    switch (type) {
        case 'success': return 'fa-check-circle';
        case 'error': return 'fa-exclamation-circle';
        case 'warning': return 'fa-exclamation-triangle';
        default: return 'fa-info-circle';
    }
}

function getNotificationColor(type) {
    switch (type) {
        case 'success': return '#28a745';
        case 'error': return '#dc3545';
        case 'warning': return '#ffc107';
        default: return '#17a2b8';
    }
}

// Close modal when clicking outside
audioModal.addEventListener('click', function(e) {
    if (e.target === audioModal) {
        closeAudioModal();
    }
});

// Add notification animation styles
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .notification-content {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .notification-close {
        background: none;
        border: none;
        color: white;
        cursor: pointer;
        padding: 2px;
        border-radius: 2px;
        margin-left: auto;
    }
    
    .notification-close:hover {
        background: rgba(255,255,255,0.2);
    }
`;
document.head.appendChild(style);

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
    console.log('MAS Planning Chat Demo initialized');
    
    // Focus on message input
    messageInput.focus();
    
    // Generate random session IDs if empty
    if (!document.getElementById('sessionId').value) {
        document.getElementById('sessionId').value = `session-${Math.random().toString(36).substr(2, 9)}`;
    }
    
    if (!document.getElementById('conversationId').value) {
        document.getElementById('conversationId').value = `conv-${Math.random().toString(36).substr(2, 9)}`;
    }
});