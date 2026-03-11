# Multimodal File Upload Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add file upload support (images, videos, audio) to the chat page for multimodal model testing.

**Architecture:** 
- Backend: Add `/v1/files/upload` endpoint to handle file uploads and return base64 data URIs
- Frontend: Add attachment button, file picker, drag-drop, and preview UI; modify message format to support multimodal content

**Tech Stack:** FastAPI (backend), Alpine.js + Tailwind CSS (frontend)

---

## Task 1: Backend - Add File Upload Endpoint

**Files:**
- Modify: `/tmp/omlx/omlx/server.py`

**Step 1: Add file upload endpoint**

Add after the existing imports (around line 50):

```python
import base64
import mimetypes
from pathlib import Path
```

Add the upload endpoint after `/v1/chat/completions` (around line 1600):

```python
# File upload configuration
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/webm", "video/quicktime"}
ALLOWED_AUDIO_TYPES = {"audio/mpeg", "audio/wav", "audio/ogg", "audio/mp4", "audio/x-m4a"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


@app.post("/v1/files/upload")
async def upload_file(
    file: UploadFile = File(...),
    _: bool = Depends(verify_api_key),
):
    """
    Upload a file (image, video, or audio) and return a base64 data URI.
    
    Returns:
        {
            "url": "data:image/jpeg;base64,...",
            "type": "image" | "video" | "audio",
            "filename": "original_filename.jpg"
        }
    """
    # Validate file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Detect content type
    content_type = file.content_type or mimetypes.guess_type(file.filename or "")[0]
    if not content_type:
        raise HTTPException(status_code=400, detail="Could not determine file type")
    
    # Categorize file type
    if content_type in ALLOWED_IMAGE_TYPES:
        file_type = "image"
    elif content_type in ALLOWED_VIDEO_TYPES:
        file_type = "video"
    elif content_type in ALLOWED_AUDIO_TYPES:
        file_type = "audio"
    else:
        allowed = ALLOWED_IMAGE_TYPES | ALLOWED_VIDEO_TYPES | ALLOWED_AUDIO_TYPES
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {content_type}. Allowed types: {allowed}"
        )
    
    # Convert to base64 data URI
    base64_data = base64.b64encode(contents).decode("utf-8")
    data_uri = f"data:{content_type};base64,{base64_data}"
    
    return {
        "url": data_uri,
        "type": file_type,
        "filename": file.filename or "uploaded_file"
    }
```

**Step 2: Verify imports exist**

Check that `UploadFile` and `File` are imported from fastapi. If not, add to imports:

```python
from fastapi import UploadFile, File
```

**Step 3: Run server to verify endpoint**

Run: `cd /tmp/omlx && python -c "from omlx.server import app; print('OK')"`
Expected: OK (no import errors)

**Step 4: Commit**

```bash
git add omlx/server.py
git commit -m "feat: add /v1/files/upload endpoint for multimodal support"
```

---

## Task 2: Frontend - Add File Upload State and Styles

**Files:**
- Modify: `/tmp/omlx/omlx/admin/templates/chat.html`

**Step 1: Add CSS styles for file upload UI**

Insert after line 331 (after the media query block, before `{% endblock %}`):

```css
    /* File upload styles */
    .file-preview-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        padding: 8px 12px;
        border-bottom: 1px solid var(--border-faint);
    }
    
    .file-preview-item {
        position: relative;
        width: 80px;
        height: 80px;
        border-radius: 8px;
        overflow: hidden;
        background: var(--bg-secondary);
        border: 1px solid var(--border-faint);
    }
    
    .file-preview-item img,
    .file-preview-item video {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    
    .file-preview-item.audio {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        gap: 4px;
    }
    
    .file-preview-item.audio svg {
        width: 32px;
        height: 32px;
        color: var(--text-tertiary);
    }
    
    .file-preview-item.audio .filename {
        font-size: 9px;
        color: var(--text-secondary);
        max-width: 70px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        text-align: center;
    }
    
    .file-remove-btn {
        position: absolute;
        top: 4px;
        right: 4px;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: rgba(0, 0, 0, 0.6);
        color: white;
        border: none;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        line-height: 1;
    }
    
    .file-remove-btn:hover {
        background: rgba(0, 0, 0, 0.8);
    }
    
    .attach-btn {
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        border: none;
        background: transparent;
        color: var(--text-tertiary);
        cursor: pointer;
        transition: all 0.15s;
        flex-shrink: 0;
    }
    
    .attach-btn:hover {
        background: var(--bg-secondary);
        color: var(--text-secondary);
    }
    
    .drop-zone {
        position: relative;
    }
    
    .drop-zone.drag-over::after {
        content: '';
        position: absolute;
        inset: 0;
        border: 2px dashed var(--text-tertiary);
        border-radius: 1.5rem;
        background: var(--bg-secondary);
        opacity: 0.5;
        pointer-events: none;
    }
```

**Step 2: Commit**

```bash
git add omlx/admin/templates/chat.html
git commit -m "feat(chat): add file upload CSS styles"
```

---

## Task 3: Frontend - Add File Upload UI Elements

**Files:**
- Modify: `/tmp/omlx/omlx/admin/templates/chat.html`

**Step 1: Add file input and modify input area**

Replace the Input Area section (lines 564-587) with:

```html
        <!-- Input Area -->
        <div class="p-4 border-t border-border-faint bg-surface-primary flex-shrink-0">
            <div class="max-w-4xl mx-auto">
                <div class="input-container drop-zone" 
                     :class="{ 'drag-over': isDragging }"
                     @dragover.prevent="isDragging = true"
                     @dragleave.prevent="isDragging = false"
                     @drop.prevent="handleDrop($event)">
                    
                    <!-- File Previews -->
                    <div x-show="pendingFiles.length > 0" class="file-preview-container">
                        <template x-for="(file, index) in pendingFiles" :key="index">
                            <div class="file-preview-item" 
                                 :class="{ 'audio': file.type === 'audio' }">
                                <template x-if="file.type === 'image'">
                                    <img :src="file.url" :alt="file.filename">
                                </template>
                                <template x-if="file.type === 'video'">
                                    <video :src="file.url" muted></video>
                                </template>
                                <template x-if="file.type === 'audio'">
                                    <div class="audio">
                                        <i data-lucide="music" class="w-8 h-8"></i>
                                        <span class="filename" x-text="file.filename"></span>
                                    </div>
                                </template>
                                <button class="file-remove-btn" @click="removeFile(index)">&times;</button>
                            </div>
                        </template>
                    </div>
                    
                    <div class="flex items-end">
                        <!-- Hidden file input -->
                        <input type="file" 
                               x-ref="fileInput"
                               @change="handleFileSelect($event)"
                               accept="image/*,video/*,audio/*"
                               multiple
                               class="hidden">
                        
                        <!-- Attach button -->
                        <button class="attach-btn" 
                                @click="$refs.fileInput.click()"
                                :disabled="!apiKeySet || !currentModel || isStreaming"
                                title="{{ t('chat.attach_file') }}">
                            <i data-lucide="paperclip" class="w-5 h-5"></i>
                        </button>
                        
                        <textarea
                            x-model="inputMessage"
                            @keydown.enter="if (!$event.shiftKey) { $event.preventDefault(); sendMessage(); }"
                            @input="autoResize($event.target)"
                            placeholder="{{ t('chat.input_placeholder') }}"
                            :disabled="!apiKeySet || !currentModel || isStreaming"
                            rows="1"
                            class="flex-1 px-2 py-3 bg-transparent resize-none outline-none text-sm max-h-32"
                            style="color: var(--text-primary);"
                        ></textarea>
                        <button
                            @click="sendMessage()"
                            :disabled="(!inputMessage.trim() && pendingFiles.length === 0) || !apiKeySet || !currentModel || isStreaming"
                            class="w-10 h-10 m-1 bg-neutral-900 hover:bg-neutral-800 disabled:bg-neutral-300 rounded-full transition-colors flex items-center justify-center flex-shrink-0"
                        >
                            <i data-lucide="arrow-up" class="w-4 h-4 text-white"></i>
                        </button>
                    </div>
                </div>
                <p class="text-xs text-center mt-2" style="color: var(--text-muted);">
                    {{ t('chat.upload_hint') }}
                </p>
            </div>
        </div>
```

**Step 2: Commit**

```bash
git add omlx/admin/templates/chat.html
git commit -m "feat(chat): add file upload UI with drag-drop support"
```

---

## Task 4: Frontend - Add File Upload JavaScript Logic

**Files:**
- Modify: `/tmp/omlx/omlx/admin/templates/chat.html`

**Step 1: Add file upload state variables**

In the `chatApp()` function, add after line 654 (after `editContent: ''`):

```javascript
            // File Upload State
            pendingFiles: [],
            isDragging: false,
            isUploading: false,
```

**Step 2: Add file upload methods**

Add these methods in the `chatApp()` object, after `applyTheme()` method (around line 1435):

```javascript
            // File upload methods
            handleFileSelect(event) {
                const files = event.target.files;
                if (files) {
                    this.processFiles(files);
                }
                event.target.value = ''; // Reset input
            },

            handleDrop(event) {
                this.isDragging = false;
                const files = event.dataTransfer.files;
                if (files) {
                    this.processFiles(files);
                }
            },

            async processFiles(files) {
                this.isUploading = true;
                
                for (const file of files) {
                    try {
                        const uploadedFile = await this.uploadFile(file);
                        this.pendingFiles.push(uploadedFile);
                    } catch (error) {
                        console.error('Upload failed:', error);
                        alert(`Failed to upload ${file.name}: ${error.message}`);
                    }
                }
                
                this.isUploading = false;
                this.$nextTick(() => lucide.createIcons());
            },

            async uploadFile(file) {
                const formData = new FormData();
                formData.append('file', file);
                
                const response = await fetch('/v1/files/upload', {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${this.getApiKey()}`
                    },
                    body: formData
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Upload failed');
                }
                
                return await response.json();
            },

            removeFile(index) {
                this.pendingFiles.splice(index, 1);
            },
```

**Step 3: Commit**

```bash
git add omlx/admin/templates/chat.html
git commit -m "feat(chat): add file upload JavaScript logic"
```

---

## Task 5: Frontend - Modify Message Sending for Multimodal

**Files:**
- Modify: `/tmp/omlx/omlx/admin/templates/chat.html`

**Step 1: Update sendMessage method**

Replace the `sendMessage()` method (around lines 833-855) with:

```javascript
            async sendMessage() {
                if ((!this.inputMessage.trim() && this.pendingFiles.length === 0) || !this.currentModel || this.isStreaming) return;

                const userMessage = this.inputMessage.trim();
                const files = [...this.pendingFiles];
                this.inputMessage = '';
                this.pendingFiles = [];

                // Create new chat if needed
                if (!this.currentChatId) {
                    this.currentChatId = 'chat_' + Date.now();
                }

                // Build message content - multimodal format
                let messageContent;
                if (files.length > 0) {
                    // Build content array for multimodal message
                    const contentParts = [];
                    
                    // Add text if present
                    if (userMessage) {
                        contentParts.push({
                            type: 'text',
                            text: userMessage
                        });
                    }
                    
                    // Add files
                    for (const file of files) {
                        if (file.type === 'image') {
                            contentParts.push({
                                type: 'image_url',
                                image_url: { url: file.url }
                            });
                        } else if (file.type === 'video') {
                            contentParts.push({
                                type: 'video_url',
                                video_url: { url: file.url }
                            });
                        } else if (file.type === 'audio') {
                            contentParts.push({
                                type: 'audio_url',
                                audio_url: { url: file.url }
                            });
                        }
                    }
                    
                    messageContent = contentParts;
                    
                    // Store for display
                    this.messages.push({
                        role: 'user',
                        content: userMessage || '[Attached files]',
                        _files: files  // For display purposes
                    });
                } else {
                    // Text-only message
                    messageContent = userMessage;
                    
                    this.messages.push({
                        role: 'user',
                        content: userMessage
                    });
                }

                // Force scroll to bottom when sending message
                this.forceScrollToBottom();

                // Start streaming with multimodal content
                await this.streamResponse(messageContent);
            },
```

**Step 2: Update streamResponse to accept message content**

Replace the `streamResponse()` method (around lines 857-976) with:

```javascript
            async streamResponse(messageContent = null) {
                this.isStreaming = true;
                this.streamingContent = '';
                this.abortController = new AbortController();
                this.autoScrollEnabled = true; // Enable auto-scroll at start of streaming

                // Build messages for API - use provided content or fall back to this.messages
                let messagesForAPI;
                if (messageContent !== null) {
                    // Use the provided content for the last user message
                    messagesForAPI = this.messages.slice(0, -1).map(msg => {
                        // Clean internal display properties
                        const { _files, ...cleanMsg } = msg;
                        return {
                            role: cleanMsg.role,
                            content: cleanMsg.content
                        };
                    });
                    messagesForAPI.push({
                        role: 'user',
                        content: messageContent
                    });
                } else {
                    // Regenerate - use existing messages
                    messagesForAPI = this.messages.map(msg => {
                        const { _files, ...cleanMsg } = msg;
                        return {
                            role: cleanMsg.role,
                            content: cleanMsg.content
                        };
                    });
                }

                try {
                    const response = await fetch('/v1/chat/completions', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${this.getApiKey()}`
                        },
                        body: JSON.stringify({
                            model: this.currentModel,
                            messages: messagesForAPI,
                            stream: true
                        }),
                        signal: this.abortController.signal
                    });

                    if (!response.ok) {
                        const errorText = await response.text();
                        throw new Error(errorText || `Error: ${response.status}`);
                    }

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let buffer = '';

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;

                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\n');
                        buffer = lines.pop() || '';

                        for (const line of lines) {
                            if (line.trim() === '' || line.trim() === 'data: [DONE]') continue;

                            if (line.startsWith('data: ')) {
                                try {
                                    const data = JSON.parse(line.slice(6));
                                    const delta = data.choices?.[0]?.delta;
                                    if (delta?.reasoning_content) {
                                        if (!this.thinkingState.isInThinking) {
                                            this.streamingContent += '<think">';
                                            this.thinkingState.isInThinking = true;
                                        }
                                        this.streamingContent += delta.reasoning_content;
                                        this.scrollToBottom();
                                    }
                                    if (delta?.content) {
                                        if (this.thinkingState.isInThinking) {
                                            this.streamingContent += '</think">';
                                            this.thinkingState.isInThinking = false;
                                        }
                                        this.streamingContent += delta.content;
                                        this.scrollToBottom();
                                    }
                                } catch (e) {
                                    console.error('Error parsing SSE:', e);
                                }
                            }
                        }
                    }

                    // Close unclosed thinking block
                    if (this.thinkingState.isInThinking) {
                        this.streamingContent += '</think">';
                        this.thinkingState.isInThinking = false;
                    }

                    // Add completed message
                    if (this.streamingContent) {
                        this.messages.push({
                            role: 'assistant',
                            content: this.streamingContent,
                            model: this.currentModel
                        });
                        this.saveCurrentChat();
                    }

                } catch (error) {
                    console.log('[streamResponse] catch error:', error.name, error.message);
                    if (error.name === 'AbortError') {
                        console.log('[streamResponse] AbortError caught - stream stopped by user');
                        if (this.thinkingState.isInThinking) {
                            this.streamingContent += '</think">';
                            this.thinkingState.isInThinking = false;
                        }
                        if (this.streamingContent) {
                            this.messages.push({
                                role: 'assistant',
                                content: this.streamingContent,
                                model: this.currentModel
                            });
                            this.saveCurrentChat();
                        }
                    } else {
                        console.error('Streaming error:', error);
                        this.messages.push({
                            role: 'assistant',
                            content: `Error: ${error.message}`,
                            model: this.currentModel
                        });
                    }
                } finally {
                    this.isStreaming = false;
                    this.streamingContent = '';
                    this.abortController = null;
                    this.thinkingAutoScroll = true;
                    this.$nextTick(() => lucide.createIcons());
                }
            },
```

**Step 3: Commit**

```bash
git add omlx/admin/templates/chat.html
git commit -m "feat(chat): modify message sending for multimodal content"
```

---

## Task 6: Frontend - Update Message Display for Files

**Files:**
- Modify: `/tmp/omlx/omlx/admin/templates/chat.html`

**Step 1: Update user message display to show attached files**

Find the user message template (around lines 492-516) and replace with:

```html
                        <!-- User Message -->
                        <div x-show="msg.role === 'user'" class="flex justify-end">
                            <div class="user-message">
                                <button @click="startEdit(index)" class="user-edit-btn" title="{{ t('chat.edit_tooltip') }}" x-show="!msg._files">
                                    <i data-lucide="pencil" class="w-4 h-4"></i>
                                </button>
                                <template x-if="editingIndex !== index">
                                    <div>
                                        <!-- Show attached files preview -->
                                        <template x-if="msg._files && msg._files.length > 0">
                                            <div class="flex flex-wrap gap-1 mb-1" style="max-width: 300px;">
                                                <template x-for="(file, fIndex) in msg._files" :key="fIndex">
                                                    <div class="relative" style="width: 60px; height: 60px;">
                                                        <template x-if="file.type === 'image'">
                                                            <img :src="file.url" class="w-full h-full object-cover rounded-lg">
                                                        </template>
                                                        <template x-if="file.type === 'video'">
                                                            <video :src="file.url" class="w-full h-full object-cover rounded-lg"></video>
                                                        </template>
                                                        <template x-if="file.type === 'audio'">
                                                            <div class="w-full h-full rounded-lg flex items-center justify-center" style="background: var(--bg-tertiary);">
                                                                <i data-lucide="music" class="w-6 h-6" style="color: var(--text-tertiary);"></i>
                                                            </div>
                                                        </template>
                                                    </div>
                                                </template>
                                            </div>
                                        </template>
                                        <div class="user-message-bubble" x-html="renderMarkdown(msg.content)"></div>
                                    </div>
                                </template>
                                <template x-if="editingIndex === index">
                                    <div class="user-message-bubble" style="padding: 0; background: transparent;">
                                        <textarea
                                            x-model="editContent"
                                            @keydown.enter.ctrl="saveEdit(index)"
                                            @keydown.escape="cancelEdit"
                                            class="inline-edit-textarea"
                                            x-init="$nextTick(() => { $el.focus(); $el.setSelectionRange($el.value.length, $el.value.length); })"
                                        ></textarea>
                                        <div class="inline-edit-actions">
                                            <button @click="cancelEdit" class="px-3 py-1.5 text-sm border rounded-lg hover:bg-neutral-100" style="border-color: var(--border-normal); color: var(--text-secondary);">{{ t('chat.edit_cancel') }}</button>
                                            <button @click="saveEdit(index)" class="px-3 py-1.5 text-sm bg-neutral-900 text-white rounded-lg hover:bg-neutral-800">{{ t('chat.edit_save') }}</button>
                                        </div>
                                    </div>
                                </template>
                            </div>
                        </div>
```

**Step 2: Commit**

```bash
git add omlx/admin/templates/chat.html
git commit -m "feat(chat): display attached files in user messages"
```

---

## Task 7: Add Translation Strings

**Files:**
- Modify: Translation files (need to find location)

**Step 1: Find translation files**

Run: `find /tmp/omlx -name "*.json" -path "*/translations/*" -o -name "*.yaml" -path "*/translations/*"`
Expected: List of translation files

**Step 2: Add translation keys**

Add to each translation file:

```json
{
  "chat": {
    "attach_file": "Attach file",
    "upload_hint": "Attach images, videos, or audio files. Drag & drop or click the paperclip.",
    ...
  }
}
```

**Step 3: Commit**

```bash
git add <translation files>
git commit -m "feat(i18n): add file upload translation strings"
```

---

## Task 8: Testing and Verification

**Step 1: Start the server**

Run: `cd /tmp/omlx && python -m omlx.server --host 0.0.0.0 --port 8000`

**Step 2: Test file upload endpoint**

Run: `curl -X POST "http://localhost:8000/v1/files/upload" -H "Authorization: Bearer <api_key>" -F "file=@/path/to/test.jpg"`
Expected: JSON response with `url`, `type`, `filename`

**Step 3: Test chat UI**

1. Open `http://localhost:8000/admin/chat`
2. Click paperclip button - file picker should open
3. Select an image - preview should appear
4. Type message and send - image should be included
5. Drag and drop a file - should upload and preview

**Step 4: Test multimodal model**

1. Load a VLM model (e.g., qwen2-vl)
2. Upload an image and ask "What's in this image?"
3. Verify model responds with image analysis

---

## Summary

**Files Modified:**
1. `/tmp/omlx/omlx/server.py` - Backend upload endpoint
2. `/tmp/omlx/omlx/admin/templates/chat.html` - Frontend UI and logic
3. Translation files (optional)

**Commits:**
1. `feat: add /v1/files/upload endpoint for multimodal support`
2. `feat(chat): add file upload CSS styles`
3. `feat(chat): add file upload UI with drag-drop support`
4. `feat(chat): add file upload JavaScript logic`
5. `feat(chat): modify message sending for multimodal content`
6. `feat(chat): display attached files in user messages`
7. `feat(i18n): add file upload translation strings` (optional)
