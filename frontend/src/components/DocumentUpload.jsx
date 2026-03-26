import { useState } from 'react'
import './DocumentUpload.css'

function DocumentUpload({ onDocumentUpload, loading: parentLoading = false }) {
  const [pastedContent, setPastedContent] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const [userPrompt, setUserPrompt] = useState('')
  const [loading, setLoading] = useState(false)
  const isLoading = loading || parentLoading

  const handleFileSelect = (e) => {
    const file = e.target.files[0]
    setSelectedFile(file || null)
  }

  const handleGenerate = async () => {
    // Prefer file if selected, otherwise use pasted text
    if (selectedFile) {
      setLoading(true)
      const formData = new FormData()
      formData.append('file', selectedFile)
      try {
        const response = await fetch('/api/documents/upload', {
          method: 'POST',
          body: formData
        })
        if (!response.ok) {
          const error = await response.json()
          alert('Error: ' + error.detail)
          return
        }
        const data = await response.json()
        onDocumentUpload(data.content, userPrompt.trim() || null)
        setSelectedFile(null)
      } catch (error) {
        alert('Error uploading file: ' + error.message)
      } finally {
        setLoading(false)
      }
    } else if (pastedContent.trim()) {
      setLoading(true)
      try {
        const response = await fetch('/api/documents/paste', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: pastedContent.trim() })
        })
        if (!response.ok) {
          const error = await response.json()
          alert('Error: ' + (error.detail || 'Unknown error'))
          setLoading(false)
          return
        }
        const data = await response.json()
        onDocumentUpload(data.content, userPrompt.trim() || null)
        setPastedContent('')
      } catch (error) {
        alert('Error: ' + error.message)
        setLoading(false)
      }
    } else {
      alert('Please upload a file or paste text first')
    }
  }

  const hasInput = selectedFile || pastedContent.trim()

  const handleShortcutGenerate = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault()
      if (!isLoading && hasInput) {
        handleGenerate()
      }
    }
  }

  return (
    <div className="document-upload">
      <div className="generation-grid">
        <div className="generation-panel upload-area compact">
          <div className="panel-title">Upload File</div>
          <div className="upload-form">
            <label className="file-input-box" htmlFor="file-upload-input">
              <div className="upload-icon">📄</div>
              <h3>Select a file to upload</h3>
              <p className="file-types">TXT, PDF, MD, DOCX, CSV</p>
              <span className="choose-file-pill">Choose File</span>
              <input
                id="file-upload-input"
                type="file"
                onChange={handleFileSelect}
                disabled={isLoading}
                accept=".txt,.pdf,.md,.docx,.csv"
              />
            </label>
            {selectedFile && <p className="selected-file">Selected: {selectedFile.name}</p>}
          </div>
        </div>

        <div className="generation-panel paste-panel compact">
          <div className="panel-title">Paste Text</div>
          <div className="paste-form">
            <textarea
              placeholder="Paste your document content here..."
              rows="8"
              disabled={isLoading}
              value={pastedContent}
              onChange={(e) => setPastedContent(e.target.value)}
              onKeyDown={handleShortcutGenerate}
            />
          </div>
        </div>
      </div>

      {/* ── Prompt Box ── */}
      <div className="prompt-section">
        <div className="prompt-panel">
          <div className="panel-title">🎯 Prompt </div>
          {/* <p className="prompt-hint">
            Tell the AI what to focus on. E.g. "Generate test cases for the login feature" or "Focus on Modbus communication settings".
          </p> */}
          <textarea
            className="prompt-input"
            placeholder='e.g. "Generate test cases for alarm monitoring feature"'
            rows="3"
            disabled={isLoading}
            value={userPrompt}
            onChange={(e) => setUserPrompt(e.target.value)}
            onKeyDown={handleShortcutGenerate}
          />
        </div>
      </div>

      {/* ── Common Generate Button ── */}
      <button
        className="generate-btn"
        onClick={handleGenerate}
        disabled={isLoading || !hasInput}
      >
        {isLoading ? 'Generating test cases...' : 'Generate Test Cases'}
      </button>

      {isLoading && <p className="loading">Generating test cases...</p>}
    </div>
  )
}

export default DocumentUpload
