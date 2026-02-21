import { useState, useRef, useEffect, useCallback } from 'react'
import { Send, Loader2 } from 'lucide-react'
import ChatMessage from './ChatMessage'
import ImageUpload from './ImageUpload'
import { sendChatMessage } from '../api'

export default function ChatSection() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hi! I can help you find products from Arcaplanet (pet supplies) and Twinset (fashion). What are you looking for?',
    },
  ])
  const [input, setInput] = useState('')
  const [pendingImage, setPendingImage] = useState(null)
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  const handleSend = async () => {
    const text = input.trim()
    if (!text && !pendingImage) return

    const userMessage = {
      role: 'user',
      content: text || '(image uploaded)',
      imagePreview: pendingImage?.preview || null,
    }

    const newMessages = [...messages, userMessage]
    setMessages(newMessages)
    setInput('')
    setLoading(true)

    const imageBase64 = pendingImage?.base64 || null
    setPendingImage(null)

    try {
      const chatHistory = newMessages
        .filter((m) => !m.imagePreview || m.content !== '(image uploaded)')
        .map((m) => ({ role: m.role, content: m.content }))

      const response = await sendChatMessage(chatHistory, imageBase64)

      const assistantMessage = {
        role: 'assistant',
        content: response.message,
        products: response.products || null,
        stores: response.stores || null,
        followUpQuestions: response.follow_up_questions || null,
        onQuestionClick: (q) => {
          setInput(q)
          inputRef.current?.focus()
        },
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (err) {
      console.error('Chat error:', err)
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `Sorry, something went wrong: ${err.message}`,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex flex-col h-[500px] border border-gray-200 rounded-xl bg-white">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        <div className="max-w-3xl mx-auto">
          {messages.map((msg, i) => (
            <ChatMessage key={i} message={msg} />
          ))}
          {loading && (
            <div className="flex justify-start mb-4">
              <div className="bg-gray-100 rounded-2xl rounded-bl-md px-4 py-3">
                <Loader2 size={18} className="animate-spin text-gray-500" />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Image preview */}
      {pendingImage && (
        <div className="px-4 pb-2">
          <div className="max-w-3xl mx-auto flex items-center gap-2">
            <img
              src={pendingImage.preview}
              alt="Upload preview"
              className="h-16 w-16 rounded-lg object-cover border border-gray-200"
            />
            <span className="text-sm text-gray-500">{pendingImage.name}</span>
            <button
              onClick={() => setPendingImage(null)}
              className="text-xs text-red-500 hover:text-red-700"
            >
              Remove
            </button>
          </div>
        </div>
      )}

      {/* Input area */}
      <div className="border-t border-gray-200 bg-white px-4 py-3 rounded-b-xl">
        <div className="max-w-3xl mx-auto flex items-end gap-2">
          <ImageUpload
            onImageSelect={setPendingImage}
            disabled={loading}
          />
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask for product recommendations..."
              disabled={loading}
              rows={1}
              className="w-full resize-none rounded-xl border border-gray-300 px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 placeholder-gray-400"
            />
          </div>
          <button
            onClick={handleSend}
            disabled={loading || (!input.trim() && !pendingImage)}
            className="p-2.5 rounded-xl bg-blue-600 text-white hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  )
}
