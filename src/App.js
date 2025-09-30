import React, { useState } from 'react';
import './App.css';

function App() {
  const [studentId, setStudentId] = useState('');
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [analysis, setAnalysis] = useState('');
  const [documentsUsed, setDocumentsUsed] = useState(0);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('chat');
  const backendUrl = 'https://rag-student-chatbot-c64a020171ba.herokuapp.com';

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!studentId || !query) return;
    setLoading(true);
    try {
      const response = await fetch(`${backendUrl}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          student_id: studentId,
          query: query
        })
      });
      const data = await response.json();
      setResponse(data.response);
      setAnalysis(data.analysis);
      setDocumentsUsed(data.documents_used);
    } catch (error) {
      setResponse('Error connecting to server');
    }
    setLoading(false);
  };

  const handleAddDataSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const questionData = {
      student_id: formData.get('student_id'),
      assessment_id: formData.get('assessment_id'),
      question: formData.get('question'),
      question_type: formData.get('question_type'),
      total_marks: parseInt(formData.get('total_marks')),
      obtained_marks: parseInt(formData.get('obtained_marks')),
      feedback: formData.get('feedback'),
      tags: formData.get('tags').split(',').map(tag => tag.trim()),
      date: formData.get('date')
    };
    try {
      const response = await fetch(`${backendUrl}/api/add_question_data`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(questionData)
      });
      const result = await response.json();
      alert(result.message);
    } catch (error) {
      alert('Error adding question data');
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Student Performance Chatbot</h1>
      </header>
      <div className="app-container">
        <div className="tabs">
          <button 
            className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >
            Chat with Bot
          </button>
          <button 
            className={`tab ${activeTab === 'addData' ? 'active' : ''}`}
            onClick={() => setActiveTab('addData')}
          >
            Add Question Data
          </button>
        </div>
        {activeTab === 'chat' && (
          <div className="chat-section">
            <form onSubmit={handleChatSubmit} className="chat-form">
              <div className="form-group">
                <label>Student ID:</label>
                <input
                  type="text"
                  value={studentId}
                  onChange={(e) => setStudentId(e.target.value)}
                  placeholder="Enter your student ID"
                  required
                />
              </div>
              <div className="form-group">
                <label>Your Question:</label>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Ask about your performance, improvement tips, etc."
                  rows="4"
                  required
                />
              </div>
              <button type="submit" disabled={loading}>
                {loading ? 'Processing...' : 'Get Advice'}
              </button>
            </form>
            {response && (
              <div className="response-section">
                <div className="response-card">
                  <h3>Advice:</h3>
                  <p>{response}</p>
                </div>
                {analysis && (
                  <div className="analysis-card">
                    <h3>Performance Analysis:</h3>
                    <p>{analysis}</p>
                  </div>
                )}
                <div className="meta-info">
                  <span>Documents analyzed: {documentsUsed}</span>
                </div>
              </div>
            )}
          </div>
        )}
        {activeTab === 'addData' && (
          <div className="add-data-section">
            <form onSubmit={handleAddDataSubmit} className="data-form">
              <div className="form-row">
                <div className="form-group">
                  <label>Student ID:</label>
                  <input type="text" name="student_id" required />
                </div>
                <div className="form-group">
                  <label>Assessment ID:</label>
                  <input type="text" name="assessment_id" required />
                </div>
              </div>
              <div className="form-group">
                <label>Question:</label>
                <textarea name="question" rows="3" required />
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label>Question Type:</label>
                  <select name="question_type" required>
                    <option value="theory">Theory</option>
                    <option value="practical">Practical</option>
                    <option value="numerical">Numerical</option>
                    <option value="diagram">Diagram</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Date:</label>
                  <input type="date" name="date" required />
                </div>
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label>Total Marks:</label>
                  <input type="number" name="total_marks" required />
                </div>
                <div className="form-group">
                  <label>Obtained Marks:</label>
                  <input type="number" name="obtained_marks" required />
                </div>
              </div>
              <div className="form-group">
                <label>Feedback:</label>
                <textarea name="feedback" rows="2" required />
              </div>
              <div className="form-group">
                <label>Tags (comma separated):</label>
                <input type="text" name="tags" placeholder="biology, diagram, theory" required />
              </div>
              <button type="submit">Add Question Data</button>
            </form>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
