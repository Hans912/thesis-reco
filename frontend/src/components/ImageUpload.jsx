import { useRef } from 'react'
import { Paperclip } from 'lucide-react'

export default function ImageUpload({ onImageSelect, disabled }) {
  const inputRef = useRef(null)

  const handleChange = (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = () => {
      const base64 = reader.result.split(',')[1]
      const preview = reader.result
      onImageSelect({ base64, preview, name: file.name })
    }
    reader.readAsDataURL(file)
    e.target.value = ''
  }

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={handleChange}
        className="hidden"
      />
      <button
        onClick={() => inputRef.current?.click()}
        disabled={disabled}
        className="p-2 rounded-lg text-gray-500 hover:text-gray-700 hover:bg-gray-100 transition-colors disabled:opacity-50"
        title="Upload image"
        type="button"
      >
        <Paperclip size={20} />
      </button>
    </>
  )
}
