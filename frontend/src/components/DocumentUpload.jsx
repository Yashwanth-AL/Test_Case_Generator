import { useState } from 'react'
import './DocumentUpload.css'

function DocumentUpload({ onDocumentUpload, loading: parentLoading = false }) {
  const [pastedContent, setPastedContent] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const isLoading = loading || parentLoading

  const handleFileSelect = (e) => {
    const file = e.target.files[0]
    setSelectedFile(file || null)
  }

  const handleGenerateFromFile = async () => {
    if (!selectedFile) {
      alert('Please select a file first')
      return
    }

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
      onDocumentUpload(data.content)
      setSelectedFile(null)
    } catch (error) {
      alert('Error uploading file: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  const handlePasteText = async () => {
    const content = pastedContent.trim()

    if (!content) {
      alert('Please paste some text')
      return
    }

    setLoading(true)
    try {
      console.log('Sending paste request...')
      const response = await fetch('/api/documents/paste', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content })
      })

      console.log('Response status:', response.status)
      
      if (!response.ok) {
        const error = await response.json()
        console.error('Error response:', error)
        alert('Error: ' + (error.detail || 'Unknown error'))
        setLoading(false)
        return
      }

      const data = await response.json()
      console.log('Success response:', data)
      onDocumentUpload(data.content)
      setPastedContent('')
    } catch (error) {
      console.error('Fetch error:', error)
      alert('Error: ' + error.message)
      setLoading(false)
    }
  }

  return (
    <div className="document-upload">
      <h2>Test Generation</h2>
      <p>Choose either method below to start generating test cases.</p>

      <div className="generation-grid">
        <div className="generation-panel upload-area">
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

          <button
            className="submit-btn"
            onClick={handleGenerateFromFile}
            disabled={isLoading || !selectedFile}
          >
            {isLoading ? 'Generating test cases...' : 'Generate Test Cases'}
          </button>
          </div>
          {isLoading && <p className="loading">Generating test cases...</p>}
        </div>

        <div className="generation-panel paste-panel">
          <div className="panel-title">Paste Text</div>
          <div className="paste-form">
          <textarea 
            placeholder="Paste your document content here..."
            rows="12"
            disabled={isLoading}
            value={pastedContent}
            onChange={(e) => setPastedContent(e.target.value)}
          />
          <button 
            className="submit-btn"
            disabled={isLoading}
            onClick={handlePasteText}
          >
            {isLoading ? 'Generating test cases...' : 'Generate Test Cases'}
          </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DocumentUpload
