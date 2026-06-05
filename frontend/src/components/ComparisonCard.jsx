export default function ComparisonCard({ comparison }) {
  return (
    <div className="bg-gray-900 border border-blue-800 rounded-xl p-5 space-y-4">
      <h3 className="font-semibold text-lg text-blue-300">
        Shot Comparison: Shot {comparison.reference_shot_id + 1} vs Shot {comparison.compare_shot_id + 1}
      </h3>

      <p className="text-gray-300 text-sm bg-gray-800 rounded-lg p-3">{comparison.summary}</p>

      {comparison.changed_fields.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs text-gray-500 uppercase tracking-wide font-semibold">What Changed</p>
          {comparison.changed_fields.map((cf, i) => {
            const improved = Math.abs(cf.cmp_val) < Math.abs(cf.ref_val)
            return (
              <div key={i} className="bg-gray-800 border border-gray-700 rounded-lg p-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium text-sm">{cf.label}</span>
                  <span className={`text-xs font-mono ${cf.delta > 0 ? 'text-orange-400' : 'text-blue-400'}`}>
                    {cf.delta > 0 ? '+' : ''}{cf.delta}°
                  </span>
                </div>
                <div className="flex gap-4 text-xs text-gray-400 mb-1">
                  <span>Reference: <span className="text-white">{cf.ref_val}°</span></span>
                  <span>Current: <span className="text-white">{cf.cmp_val}°</span></span>
                </div>
                <p className="text-sm text-gray-300">{cf.note}</p>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
