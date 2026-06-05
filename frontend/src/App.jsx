import { useState } from 'react'
import UploadPage from './pages/UploadPage'
import ResultsPage from './pages/ResultsPage'

export default function App() {
  const [jobId, setJobId] = useState(null)

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-center mb-2 text-white">
        Archery Shot Analyzer
      </h1>
      <p className="text-center text-gray-400 mb-8 text-sm">
        Upload a recurve archery video for AI-powered form analysis
      </p>

      {!jobId ? (
        <UploadPage onJobStarted={setJobId} />
      ) : (
        <ResultsPage jobId={jobId} onReset={() => setJobId(null)} />
      )}
    </div>
  )
}
