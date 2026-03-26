import { useState } from 'react'
import './App.css'
import DocumentUpload from './components/DocumentUpload'
import TestCaseDisplay from './components/TestCaseDisplay'

function App() {
  const [documentContent, setDocumentContent] = useState('')
  const [testCases, setTestCases] = useState([])
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('upload')

  const handleGenerateTestCases = async (content) => {
    setDocumentContent(content)
    setLoading(true)
    try {
      const response = await fetch('/api/test-cases/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          document_content: content,
          detail_level: 'Detailed',
          num_test_cases: 15
        })
      })

      if (!response.ok) {
        const error = await response.json()
        alert('Error: ' + error.detail)
        return
      }

      const data = await response.json()
      setTestCases(data.test_cases)
      setActiveTab('results')
    } catch (error) {
      alert('Error generating test cases: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  const downloadCSV = () => {
    if (testCases.length === 0) return

    const headers = ['ID', 'Type', 'Title', 'Method', 'Priority', 'Description', 'Steps', 'Expected Results', 'Final Result']
    const rows = testCases.map(tc => [
      tc.id,
      tc.test_type,
      tc.title,
      tc.method || 'Manual Testing',
      tc.priority,
      tc.description,
      tc.steps.join('; '),
      tc.expected_results.join('; '),
      tc.final_result || ''
    ])

    const csv = [headers, ...rows].map(row => 
      row.map(cell => `"${String(cell).replace(/"/g, '""')}"`).join(',')
    ).join('\n')

    const blob = new Blob([csv], { type: 'text/csv' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'test-cases.csv'
    a.click()
  }

  const downloadJSON = () => {
    if (testCases.length === 0) return

    const json = JSON.stringify(testCases, null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'test-cases.json'
    a.click()
  }

  return (
    <div className="app">
      <header className="top-header">
        <div className="header-inner">
          <div className="project-brand">
            <h1>AI Test Case Generator</h1>
            <p>Generate structured test cases from docs in seconds</p>
          </div>
          <div className="tabs">
            <button 
              className={`tab ${activeTab === 'upload' ? 'active' : ''}`}
              onClick={() => setActiveTab('upload')}
            >
              📄 Test Generation
            </button>
            <button 
              className={`tab ${activeTab === 'results' ? 'active' : ''}`}
              onClick={() => setActiveTab('results')}
              disabled={testCases.length === 0}
            >
              📋 Results
            </button>
          </div>
        </div>
      </header>

      <div className="container">
        <div className="main-content">
          <div className="tab-content">
            {activeTab === 'upload' && (
              <DocumentUpload onDocumentUpload={handleGenerateTestCases} loading={loading} />
            )}

            {activeTab === 'results' && (
              <TestCaseDisplay
                testCases={testCases}
                onDownloadCSV={downloadCSV}
                onDownloadJSON={downloadJSON}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
