import { useState, useRef } from 'react'

export default function UploadPage({ onJobStarted }) {
  const [file, setFile] = useState(null)
  const [drawSide, setDrawSide] = useState('right')
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef()

  const handleUpload = async () => {
    if (!file) return
    setUploading(true)
    setError(null)

    const form = new FormData()
    form.append('video', file)
    form.append('draw_side', drawSide)

    try {
      const res = await fetch('/api/analyze', { method: 'POST', body: form })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      onJobStarted(data.job_id)
    } catch (e) {
      setError(e.message)
      setUploading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div
        className="border-2 border-dashed border-gray-600 rounded-xl p-12 text-center cursor-pointer hover:border-blue-500 transition-colors"
        onClick={() => inputRef.current.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={e => setFile(e.target.files[0])}
        />
        {file ? (
          <p className="text-green-400 font-medium">{file.name}</p>
        ) : (
          <p className="text-gray-400">Click to select a video file</p>
        )}
      </div>

      <div className="flex items-center gap-4">
        <label className="text-sm text-gray-400">Draw hand:</label>
        <select
          value={drawSide}
          onChange={e => setDrawSide(e.target.value)}
          className="bg-gray-800 border border-gray-600 rounded px-3 py-1 text-sm text-white"
        >
          <option value="right">Right</option>
          <option value="left">Left</option>
        </select>
      </div>

      {error && <p className="text-red-400 text-sm">{error}</p>}

      <button
        onClick={handleUpload}
        disabled={!file || uploading}
        className="w-full py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg font-semibold transition-colors"
      >
        {uploading ? 'Uploading...' : 'Analyze Video'}
      </button>
    </div>
  )
}
