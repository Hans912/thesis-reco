import ProductCarousel from './ProductCarousel'
import StoreMap from './StoreMap'

export default function ChatMessage({ message }) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-[85%] ${isUser ? 'order-1' : ''}`}>
        <div
          className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap ${
            isUser
              ? 'bg-blue-600 text-white rounded-br-md'
              : 'bg-gray-100 text-gray-900 rounded-bl-md'
          }`}
        >
          {message.content}
        </div>

        {message.imagePreview && (
          <div className="mt-2 rounded-lg overflow-hidden max-w-xs">
            <img src={message.imagePreview} alt="Uploaded" className="w-full" />
          </div>
        )}

        {message.products && message.products.length > 0 && (
          <div className="mt-3">
            <ProductCarousel products={message.products} />
          </div>
        )}

        {message.stores && message.stores.length > 0 && (
          <StoreMap stores={message.stores} />
        )}

        {message.followUpQuestions && message.followUpQuestions.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-2">
            {message.followUpQuestions.map((q, i) => (
              <button
                key={i}
                onClick={() => message.onQuestionClick?.(q)}
                className="px-3 py-1.5 rounded-full text-xs font-medium bg-blue-50 text-blue-700 hover:bg-blue-100 transition-colors border border-blue-200"
              >
                {q}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
