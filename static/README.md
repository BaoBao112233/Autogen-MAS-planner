# MAS Planning Chat Demo UI

## Tổng quan

Đây là UI demo đơn giản để test endpoint `/ai/chat/text` với tính năng Text-to-Speech sử dụng ElevenLabs API.

## Tính năng

### 🗣️ **Chat Interface**
- Giao diện chat đẹp mắt và responsive
- Hiển thị tin nhắn của user và bot
- Animation mượt mà khi gửi/nhận tin nhắn

### 🎵 **Audio Features**
- **Text-to-Speech**: Tự động tạo file audio từ response của bot
- **Audio Player**: Phát audio trực tiếp trong chat
- **Download Audio**: Tải file audio về máy định dạng .wav
- **Audio Modal**: Popup để xem chi tiết và điều khiển audio

### ⚙️ **Configuration**
- **Session ID**: Quản lý phiên chat
- **Conversation ID**: Theo dõi cuộc hội thoại
- **Token**: Authentication token

## Cách sử dụng

### 1. **Khởi chạy server**
```bash
cd /home/baobao/Projects/MAS-Planning
python main.py
```

### 2. **Truy cập UI**
Mở browser và truy cập: `http://localhost:9000`

### 3. **Demo chat**
1. Nhập tin nhắn vào ô input
2. Nhấn Enter hoặc nút Send
3. Chờ bot response (có loading indicator)
4. Khi có response, sẽ xuất hiện các nút audio:
   - **Phát Audio**: Phát audio ngay trong chat
   - **Xem Audio**: Mở popup với audio player đầy đủ
   - **Tải về**: Download file audio .wav

### 4. **Cấu hình**
- Thay đổi Session ID, Conversation ID, Token theo nhu cầu
- Giá trị mặc định sẽ được tự động generate

## Cấu trúc Files

```
static/
├── index.html      # Giao diện chính
├── style.css       # Styling CSS
└── script.js       # JavaScript logic
```

### **index.html**
- Giao diện HTML cơ bản
- Chat container với messages area
- Input form với config panel
- Audio modal popup

### **style.css**
- Modern gradient design
- Responsive layout
- Beautiful animations
- Modal styling
- Mobile-friendly

### **script.js**
- Chat functionality
- API integration với `/ai/chat/text`
- Audio player controls
- File download logic
- Notification system
- Error handling

## API Integration

### **Endpoint**: `POST /ai/chat/text`

**Request:**
```json
{
    "conversationId": "conv-xxx",
    "sessionId": "session-xxx", 
    "token": "token-xxx",
    "message": "User message text"
}
```

**Response:**
```json
{
    "sessionId": "session-xxx",
    "response": "Bot response text",
    "error_status": "success",
    "audio_file_url": "/ai/download/audio/response_xxx.wav"
}
```

## Features chi tiết

### 🎨 **UI/UX**
- **Gradient Background**: Modern purple gradient
- **Glass Effect**: Semi-transparent chat container
- **Smooth Animations**: Fade in effects cho messages
- **Responsive Design**: Hoạt động tốt trên mobile
- **Font Awesome Icons**: Beautiful icons throughout

### 🔊 **Audio Controls**
- **Inline Player**: Phát audio trực tiếp trong message
- **Modal Player**: Popup với controls đầy đủ
- **Download**: One-click download với tên file tự động
- **Error Handling**: Graceful error handling cho audio issues

### 📱 **Responsive**
- Mobile-first design
- Adaptive layout
- Touch-friendly buttons
- Optimized for small screens

### 🚨 **Notifications**
- Success/Error notifications
- Auto-dismiss after 5 seconds
- Beautiful slide-in animations
- Different colors for different types

## Troubleshooting

### **Audio không phát được**
- Kiểm tra browser có hỗ trợ audio
- Kiểm tra file audio URL có accessible
- Kiểm tra ElevenLabs API

### **UI không load**
- Kiểm tra server đang chạy
- Kiểm tra static files đã mount đúng
- Kiểm tra browser console for errors

### **API lỗi**
- Kiểm tra network requests trong DevTools
- Verify endpoint `/ai/chat/text` hoạt động
- Kiểm tra request payload format

## Customization

### **Thay đổi theme colors**
Sửa trong `style.css`:
```css
/* Main gradient */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* Primary color */
background: #667eea;
```

### **Thêm tính năng**
- Modify `script.js` cho logic mới
- Update `index.html` cho UI elements
- Extend API integration

## Next Steps

1. **Voice Input**: Thêm speech-to-text input
2. **Audio History**: Lưu history audio files
3. **Multiple Voices**: Cho phép chọn voice khác nhau
4. **Real-time**: WebSocket cho real-time chat
5. **File Upload**: Upload audio files cho `/chat/audio` endpoint