import { MessageSquare } from 'lucide-react'
import CFSection from '../components/CFSection'
import ChatSection from '../components/ChatSection'

export default function RecommenderPage() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-6 space-y-6">
      {/* Top: CF store recommendations */}
      <CFSection />

      {/* Bottom: Chatbot */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <MessageSquare size={18} className="text-blue-600" />
          <h2 className="text-lg font-semibold text-gray-900">Product Search (Content-Based)</h2>
        </div>
        <p className="text-xs text-gray-500 mb-3">
          Ask for product recommendations using text or images. Powered by CLIP multimodal embeddings.
        </p>
        <ChatSection />
      </div>
    </div>
  )
}
