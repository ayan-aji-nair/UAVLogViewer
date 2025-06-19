<template>
    <div :id="getDivName()"
         v-bind:style="{width: width + 'px', height: height + 'px', top: top + 'px', left: left + 'px'}">
        <div id="paneContent">
            <span style="float: right; margin: 3px; cursor: pointer;" @click="close()"> X </span>
            <div class="chat-messages">
                <ul>
                    <li v-for="msg in messages" :key="msg.id" :class="['message', msg.role]">
                        <span class="timestamp">{{ formatTimestamp(msg.timestamp) }}</span>
                        <div class="message-content">{{ msg.content }}</div>
                    </li>
                </ul>
            </div>
            <div class="chat-input">
                <input type="text" v-model="newMessage" @keyup.enter="sendMessage" placeholder="Type your message...">
                <button @click="sendMessage" :disabled="!newMessage.trim()">
                    <i class="fas fa-paper-plane"></i>
                </button>
            </div>
        </div>
    </div>
</template>

<script>
import { store } from '../Globals.js'
import { baseWidget } from './baseWidget'

export default {
    name: 'ChatWidget',
    mixins: [baseWidget],
    data () {
        return {
            name: 'ChatWidget',
            state: store,
            width: 300,
            height: 400,
            left: 310,
            top: 0,
            messages: [],
            newMessage: '',
            sessionId: null
        }
    },
    created () {
        // Load existing session or create new one
        this.loadSession()
    },
    methods: {
        loadSession () {
            // Try to load existing session from localStorage
            const savedSession = localStorage.getItem('chatSession')
            if (savedSession) {
                const { sessionId, messages } = JSON.parse(savedSession)
                this.sessionId = sessionId
                this.messages = messages
            } else {
                // Create new session if none exists
                this.sessionId = this.generateSessionId()
                this.messages = []
            }
        },
        saveSession () {
            // Save current session to localStorage
            localStorage.setItem('chatSession', JSON.stringify({
                sessionId: this.sessionId,
                messages: this.messages
            }))
        },
        async sendMessage () {
            if (!this.newMessage.trim()) return

            // Add user message to local state
            const userMessage = {
                id: Date.now(),
                role: 'user',
                content: this.newMessage,
                timestamp: new Date()
            }
            this.messages.push(userMessage)
            this.saveSession() // Save after adding user message

            // Clear input
            const messageToSend = this.newMessage
            this.newMessage = ''

            try {
                // Get API URL from environment or use default
                const apiUrl = process.env.VUE_APP_API_URL || 'http://localhost:8000'
                // Send to backend
                const response = await fetch(`${apiUrl}/api/chat/message`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        message: messageToSend,
                        sessionId: this.sessionId,
                        contextData: this.getContextData()
                    })
                })

                if (!response.ok) {
                    const errorText = await response.text()
                    throw new Error(`HTTP ${response.status}: ${errorText}`)
                }

                const data = await response.json()
                console.log('Backend response:', data) // Debug log
                // Add bot response to messages
                this.messages.push({
                    id: Date.now(),
                    role: 'assistant',
                    content: data.message || 'No response received',
                    timestamp: new Date()
                })
                this.saveSession() // Save after adding bot response

                // Scroll to bottom
                this.$nextTick(() => {
                    this.scrollToBottom()
                })
            } catch (error) {
                console.error('Error sending message:', error)
                // Add error message to chat
                this.messages.push({
                    id: Date.now(),
                    role: 'system',
                    content: `Error: ${error.message}`,
                    timestamp: new Date()
                })
                this.saveSession() // Save after adding error message
            }
        },
        formatTimestamp (timestamp) {
            const date = new Date(timestamp)
            return date.toLocaleTimeString()
        },
        scrollToBottom () {
            const container = this.$el.querySelector('.chat-messages')
            if (container) {
                container.scrollTop = container.scrollHeight
            }
        },
        generateSessionId () {
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
                const r = Math.random() * 16 | 0
                const v = c === 'x' ? r : (r & 0x3 | 0x8)
                return v.toString(16)
            })
        },
        getContextData () {
            // Add any relevant context data from the application state
            return {
                // Add relevant context data here
            }
        },
        clearChat () {
            // Clear chat history
            this.messages = []
            this.sessionId = this.generateSessionId()
            this.saveSession()
        }
    },
    watch: {
        messages: {
            handler () {
                this.scrollToBottom()
            },
            deep: true
        }
    }
}
</script>

<style scoped>
    div#paneChatWidget {
        min-width: 300px;
        min-height: 400px;
        position: absolute;
        background: rgba(253, 254, 255, 0.856);
        color: #141924;
        font-size: 11px;
        font-weight: 600;
        z-index: 10000;
        box-shadow: 9px 9px 3px -6px rgba(26, 26, 26, 0.699);
        border-radius: 5px;
        user-select: none;
    }

    div#paneContent {
        height: 100%;
        display: flex;
        flex-direction: column;
    }

    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 10px;
    }

    .chat-messages ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }

    .message {
        margin-bottom: 10px;
        padding: 8px 12px;
        border-radius: 8px;
        max-width: 80%;
    }

    .message.user {
        background-color: #007bff;
        color: white;
        margin-left: auto;
    }

    .message.assistant {
        background-color: #e9ecef;
        color: #141924;
        margin-right: auto;
    }

    .message.system {
        background-color: #dc3545;
        color: white;
        margin: 0 auto;
        text-align: center;
    }

    .timestamp {
        font-size: 9px;
        opacity: 0.7;
        display: block;
        margin-bottom: 4px;
    }

    .chat-input {
        padding: 10px;
        border-top: 1px solid #dee2e6;
        display: flex;
        gap: 8px;
    }

    .chat-input input {
        flex: 1;
        padding: 8px;
        border: 1px solid #ced4da;
        border-radius: 4px;
        font-size: 11px;
    }

    .chat-input button {
        padding: 8px 12px;
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }

    .chat-input button:disabled {
        background-color: #6c757d;
        cursor: not-allowed;
    }

    .chat-input button:hover:not(:disabled) {
        background-color: #0056b3;
    }
</style>
