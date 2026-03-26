import { useState } from 'react'
import './TestCaseDisplay.css'

function TestCaseDisplay({ testCases, onDownloadCSV, onDownloadJSON }) {
  const [expandedIds, setExpandedIds] = useState([])
  const [view, setView] = useState('index')

  const getPriorityColor = (priority) => {
    switch(priority) {
      case 'Critical': return '#ff4757'
      case 'High': return '#ff9800'
      case 'Medium': return '#ffc107'
      case 'Low': return '#4caf50'
      default: return '#999'
    }
  }

  const toggleExpand = (id) => {
    setExpandedIds((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    )
  }

  const expandAll = () => setExpandedIds(testCases.map((tc) => tc.id))
  const collapseAll = () => setExpandedIds([])
  const allExpanded = testCases.length > 0 && expandedIds.length === testCases.length

  const hasMeaningfulList = (list) =>
    Array.isArray(list) && list.length > 0 && list.some((item) => item && item.trim().length > 0)

  return (
    <div className="test-case-display">
      <div className="display-header">
        <h2>Generated Test Cases ({testCases.length})</h2>
        <div className="header-actions">
          <div className="result-downloads">
            <button className="result-action-btn" onClick={onDownloadCSV}>Download CSV</button>
            <button className="result-action-btn" onClick={onDownloadJSON}>Download JSON</button>
          </div>
          <div className="view-toggle">
            <button className={`view-btn ${view === 'index' ? 'active' : ''}`} onClick={() => setView('index')}>Index View</button>
            <button className={`view-btn ${view === 'details' ? 'active' : ''}`} onClick={() => setView('details')}>Detail View</button>
          </div>
        </div>
      </div>

      {testCases.length === 0 ? (
        <div className="empty-state"><p>No test cases generated yet. Upload a document to generate.</p></div>
      ) : view === 'index' ? (
        <div className="index-table-container">
          <h3>1. Test Case Index</h3>
          <table className="index-table">
            <thead>
              <tr>
                <th>S.No</th>
                <th>Test Case ID</th>
                <th>Title</th>
                <th>Type</th>
                <th>Priority</th>
              </tr>
            </thead>
            <tbody>
              {testCases.map((tc, idx) => (
                <tr key={idx} onClick={() => { setView('details'); setExpandedIds([tc.id]) }}>
                  <td className="tc-index">{idx + 1}</td>
                  <td className="tc-id">{tc.id}</td>
                  <td>{tc.title}</td>
                  <td><span className="type-badge">{tc.test_type}</span></td>
                  <td><span className="priority-badge" style={{ backgroundColor: getPriorityColor(tc.priority) }}>{tc.priority}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="test-cases-list">
          <div className="details-header">
            <h3>2. Test Case Details</h3>
            <button className="expand-all-btn" onClick={allExpanded ? collapseAll : expandAll}>
              {allExpanded ? 'Collapse All' : 'Expand All'}
            </button>
          </div>
          {testCases.map((testCase, index) => {
            return (
            <div key={index} className="test-case-row">
              <div className="tc-serial-outside">{index + 1}</div>
              <div className="test-case-card">
                <div className="test-case-header" onClick={() => toggleExpand(testCase.id)}>
                  <div className="expand-toggle">{expandedIds.includes(testCase.id) ? '▼' : '▶'}</div>
                  <div className="test-id">{testCase.id}</div>
                  <span className="test-type">{testCase.test_type}</span>
                  <span className="priority" style={{ backgroundColor: getPriorityColor(testCase.priority) }}>{testCase.priority}</span>
                </div>

                <div className="test-case-title">{testCase.title}</div>

                {expandedIds.includes(testCase.id) && (
                  <div className="test-case-content">
                    <div className="tc-meta">
                      <span><strong>Test Case ID:</strong> {testCase.id}</span>
                      <span><strong>Type:</strong> {testCase.test_type}</span>
                      <span><strong>Method:</strong> {testCase.method || 'Manual Testing'}</span>
                    </div>

                    <div className="section">
                      <h4>Description</h4>
                      <p>{testCase.description}</p>
                    </div>

                    <div className="section">
                      <h4>Objective</h4>
                      <p>{testCase.objective}</p>
                    </div>

                    {/* Show Pre-Conditions only if meaningful */}
                    {hasMeaningfulList(testCase.preconditions) && (
                      <div className="section">
                        <h4>Pre-Conditions</h4>
                        <ul>
                          {testCase.preconditions.map((condition, idx) => (
                            <li key={idx}>{condition}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Show Steps & Expected Results only if meaningful */}
                    {hasMeaningfulList(testCase.steps) && (
                      <div className="section">
                        <h4>Steps & Expected Results</h4>
                        <div className="steps-results">
                          {testCase.steps.map((step, idx) => (
                            <div key={idx} className="step-row">
                              <div className="step-num">{idx + 1}</div>
                              <div className="step-text">{step}</div>
                              <div className="step-result">
                                {testCase.expected_results[idx] ? `→ ${testCase.expected_results[idx]}` : ''}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {hasMeaningfulList(testCase.acceptance_criteria) && (
                      <div className="section">
                        <h4>Acceptance Criteria</h4>
                        <ul>
                          {testCase.acceptance_criteria.map((criteria, idx) => (
                            <li key={idx}>{criteria}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {testCase.final_result && (
                      <div className="section final-result">
                        <h4>Final Result</h4>
                        <p>{testCase.final_result}</p>
                      </div>
                    )}

                  </div>
                )}
              </div>
            </div>
          )})}
        </div>
      )}
    </div>
  )
}

export default TestCaseDisplay
