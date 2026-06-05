const SEVERITY_STYLES = {
  significant: 'bg-red-900/40 border-red-700 text-red-300',
  moderate: 'bg-orange-900/40 border-orange-700 text-orange-300',
  minor: 'bg-yellow-900/30 border-yellow-700 text-yellow-300',
}

const SEVERITY_BADGE = {
  significant: 'bg-red-700 text-white',
  moderate: 'bg-orange-600 text-white',
  minor: 'bg-yellow-600 text-white',
}

export default function ShotCard({ shot }) {
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-lg">Shot {shot.shot_id + 1}</h3>
        <span className="text-sm text-gray-400">
          Consistency: {' '}
          <span className={shot.consistency_score >= 80 ? 'text-green-400' : shot.consistency_score >= 50 ? 'text-yellow-400' : 'text-red-400'}>
            {shot.consistency_score}%
          </span>
        </span>
      </div>

      <p className="text-gray-300 text-sm bg-gray-800 rounded-lg p-3">{shot.summary}</p>

      {shot.issues.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs text-gray-500 uppercase tracking-wide font-semibold">Form Issues</p>
          {shot.issues.map((issue, i) => (
            <div key={i} className={`border rounded-lg p-3 ${SEVERITY_STYLES[issue.severity]}`}>
              <div className="flex items-center gap-2 mb-1">
                <span className={`text-xs px-2 py-0.5 rounded font-semibold ${SEVERITY_BADGE[issue.severity]}`}>
                  {issue.severity}
                </span>
                <span className="font-medium text-sm">{issue.label}</span>
                <span className="ml-auto text-xs opacity-70">
                  {issue.measured}° (ideal: {issue.ideal}°, Δ{issue.deviation > 0 ? '+' : ''}{issue.deviation}°)
                </span>
              </div>
              <p className="text-sm opacity-90">{issue.feedback}</p>
            </div>
          ))}
        </div>
      )}

      {shot.consistency_notes.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs text-gray-500 uppercase tracking-wide font-semibold">Stability</p>
          {shot.consistency_notes.map((note, i) => (
            <p key={i} className="text-sm text-yellow-300 bg-yellow-900/20 border border-yellow-800 rounded px-3 py-2">{note}</p>
          ))}
        </div>
      )}

      {shot.full_draw_angles && (
        <details className="text-sm">
          <summary className="cursor-pointer text-gray-500 hover:text-gray-300 text-xs uppercase tracking-wide">
            Raw Angles
          </summary>
          <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-gray-400">
            {Object.entries(shot.full_draw_angles)
              .filter(([, v]) => v !== null)
              .map(([k, v]) => (
                <div key={k} className="flex justify-between bg-gray-800 rounded px-2 py-1">
                  <span className="opacity-70">{k.replace(/_/g, ' ')}</span>
                  <span className="font-mono">{typeof v === 'number' ? v.toFixed(2) : v}</span>
                </div>
              ))}
          </div>
        </details>
      )}
    </div>
  )
}
