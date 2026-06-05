import { useEffect, useState } from 'react'
import ShotCard from '../components/ShotCard'
import ComparisonCard from '../components/ComparisonCard'

const POLL_INTERVAL = 2000

export default function ResultsPage({ jobId, onReset }) {
  const [result, setResult] = useState(null)
  const [polling, setPolling] = useState(true)

  useEffect(() => {
    if (!polling) return
    const id = setInterval(async () => {
      try {
        const res = await fetch(`/api/results/${jobId}`)
        if (res.status === 404) return
        const data = await res.json()
        if (data.status === 'done' || data.status === 'error') {
          setResult(data)
          setPolling(false)
        }
      } catch {}
    }, POLL_INTERVAL)
    return () => clearInterval(id)
  }, [jobId, polling])

  if (!result) {
    return (
      <div className="text-center py-20 space-y-4">
        <div className="animate-spin w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full mx-auto" />
        <p className="text-gray-400">Analyzing video... this may take a minute.</p>
      </div>
    )
  }

  if (result.status === 'error') {
    return (
      <div className="space-y-4">
        <div className="bg-red-900/40 border border-red-700 rounded-lg p-4 text-red-300">
          <p className="font-semibold mb-1">Analysis failed</p>
          <p className="text-sm">{result.error}</p>
        </div>
        <button onClick={onReset} className="text-blue-400 hover:underline text-sm">
          Try another video
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold">
            {result.shot_count} shot{result.shot_count !== 1 ? 's' : ''} detected
          </h2>
          <p className="text-gray-400 text-sm">
            {result.video_info?.width}x{result.video_info?.height} &middot;{' '}
            {result.video_info?.fps?.toFixed(0)} fps
          </p>
        </div>
        <div className="flex gap-3">
          {result.annotated_video_path && (
            <a
              href={`/api/video/${jobId}`}
              className="px-4 py-2 bg-green-700 hover:bg-green-600 rounded-lg text-sm font-medium"
              download
            >
              Download Annotated Video
            </a>
          )}
          <button
            onClick={onReset}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm"
          >
            New Video
          </button>
        </div>
      </div>

      {result.shots.map(shot => (
        <ShotCard key={shot.shot_id} shot={shot} />
      ))}

      {result.comparison && <ComparisonCard comparison={result.comparison} />}
    </div>
  )
}
