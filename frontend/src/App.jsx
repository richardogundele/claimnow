import React, { useState } from 'react';

function App() {
    const [file, setFile] = useState(null);
    const [isDragging, setIsDragging] = useState(false);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [result, setResult] = useState(null);

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            setFile(e.dataTransfer.files[0]);
        }
    };

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files.length > 0) {
            setFile(e.target.files[0]);
        }
    };

    const handleAnalyze = async () => {
        if (!file) return;

        setIsAnalyzing(true);

        const formData = new FormData();
        formData.append('file', file);

        try {
            // Determine API URL (default to local FastAPI port 8000 or relative proxy)
            const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';

            const response = await fetch(`${apiUrl}/analyze/upload`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Server error: ${response.status}`);
            }

            const data = await response.json();

            // Map FastAPI response data to the structure the React app expects
            setResult({
                decision: data.verdict,
                confidence: data.confidence * 100, // Assuming 0-1 scale from backend
                summary: data.summary,
                extractedData: {
                    vehicleClass: data.extracted_claim?.vehicle?.group || 'Unspecified',
                    startDate: data.extracted_claim?.hire_period?.start_date || 'Unknown',
                    endDate: data.extracted_claim?.hire_period?.end_date || 'Unknown',
                    dailyRate: data.extracted_claim?.rates?.daily_rate?.toFixed(2) || '0.00'
                },
                marketRates: {
                    low: data.rate_match?.statistics?.min_rate?.toFixed(2) || '0.00',
                    high: data.rate_match?.statistics?.max_rate?.toFixed(2) || '0.00',
                    average: data.rate_match?.statistics?.mean_rate?.toFixed(2) || '0.00'
                },
                explanation: data.explanation?.summary
                    ? [data.explanation.summary, ...(data.explanation.key_factors || [])]
                    : [data.summary] // Fallback if no detailed SHAP explanation available
            });
        } catch (error) {
            console.error('Analysis failed:', error);
            alert(`Analysis failed: ${error.message}`);
        } finally {
            setIsAnalyzing(false);
        }
    };

    const resetAnalysis = () => {
        setFile(null);
        setResult(null);
    };

    return (
        <div className="container">
            <header className="app-header fade-in">
                <div className="logo-text">
                    <div className="logo-dot"></div>
                    Claims<span className="text-gradient">NOW</span>
                </div>
                <div className="d-flex align-center gap-2">
                    <span className="badge badge-fair">AWS Bedrock Engine</span>
                    <button className="btn btn-outline" style={{ padding: '0.4rem 1rem', fontSize: '0.85rem' }}>
                        Settings
                    </button>
                </div>
            </header>

            <main>
                {!result ? (
                    <div className="fade-in delay-1" style={{ maxWidth: '900px', margin: '0 auto', marginTop: '5vh' }}>
                        <div className="glass-panel text-center mb-4">
                            <h1 style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>
                                Intelligent <span className="text-gradient">Court Pack</span> Analysis
                            </h1>
                            <p style={{ fontSize: '1.2rem', marginBottom: '3rem' }}>
                                Upload motorized insurance court packs. Our AI pipeline powered by AWS Bedrock extracts details, compares with market rates, and scores fairness automatically.
                            </p>

                            <div
                                className="glass-panel mb-4"
                                style={{
                                    border: isDragging ? '2px dashed var(--accent-primary)' : '2px dashed var(--glass-border)',
                                    background: isDragging ? 'rgba(59, 130, 246, 0.05)' : 'var(--glass-bg)',
                                    cursor: 'pointer',
                                    transition: 'all 0.3s ease',
                                    padding: '4rem 2rem'
                                }}
                                onDragOver={handleDragOver}
                                onDragLeave={handleDragLeave}
                                onDrop={handleDrop}
                                onClick={() => document.getElementById('file-upload').click()}
                            >
                                <input
                                    type="file"
                                    id="file-upload"
                                    hidden
                                    accept=".pdf,.png,.jpg,.jpeg"
                                    onChange={handleFileChange}
                                />
                                <div style={{ fontSize: '3rem', marginBottom: '1rem', opacity: 0.7 }}>📄</div>
                                {file ? (
                                    <div>
                                        <h3 style={{ color: 'var(--accent-primary)', marginBottom: '0.5rem' }}>{file.name}</h3>
                                        <p>{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                                    </div>
                                ) : (
                                    <div>
                                        <h3>Drag & Drop your court pack PDF here</h3>
                                        <p style={{ marginTop: '0.5rem' }}>or click to browse from your computer</p>
                                    </div>
                                )}
                            </div>

                            {file && (
                                <button
                                    className="btn btn-primary fade-in delay-2 mt-4"
                                    style={{ width: '100%', padding: '1rem' }}
                                    onClick={handleAnalyze}
                                    disabled={isAnalyzing}
                                >
                                    {isAnalyzing ? (
                                        <span className="d-flex align-center gap-2 justify-center">
                                            <span
                                                style={{
                                                    width: '20px',
                                                    height: '20px',
                                                    border: '2px solid rgba(255,255,255,0.3)',
                                                    borderTopColor: 'white',
                                                    borderRadius: '50%',
                                                    animation: 'spin 1s linear infinite'
                                                }}
                                            />
                                            Analyzing via AWS Bedrock Pipeline...
                                        </span>
                                    ) : (
                                        'Analyze Claim Pack Now'
                                    )}
                                </button>
                            )}
                        </div>

                        {!file && (
                            <div className="workflow-section fade-in delay-3">
                                <h2>How <span className="text-gradient">ClaimsNOW</span> Works</h2>
                                <div className="timeline">
                                    <div className="timeline-item">
                                        <div className="timeline-icon">1</div>
                                        <div className="timeline-content">
                                            <h3>Document Parsing <span className="badge-tech">PyMuPDF</span></h3>
                                            <p>Extracts text and layout from motorized insurance court pack PDFs securely on your local device—no cloud OCR required.</p>
                                        </div>
                                    </div>
                                    <div className="timeline-item">
                                        <div className="timeline-icon">2</div>
                                        <div className="timeline-content">
                                            <h3>Field Extraction <span className="badge-tech">Claude Haiku 4.5</span></h3>
                                            <p>AWS Bedrock Claude Haiku 4.5 (with hybrid reasoning) extracts critical unstructured data points like vehicle class, start/end dates, and claimed rates.</p>
                                        </div>
                                    </div>
                                    <div className="timeline-item">
                                        <div className="timeline-icon">3</div>
                                        <div className="timeline-content">
                                            <h3>Market Rate Matching <span className="badge-tech">RAG + ChromaDB</span></h3>
                                            <p>Semantic search through millions of embedded rate records to find the most comparable market rates for the specific region.</p>
                                        </div>
                                    </div>
                                    <div className="timeline-item">
                                        <div className="timeline-icon">4</div>
                                        <div className="timeline-content">
                                            <h3>Verdict & Explanation <span className="badge-tech">Scikit-learn + SHAP</span></h3>
                                            <p>Custom ML classifier scores the fairness of the claim, providing an interpretible verdict backed by regulator-friendly SHAP values.</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                ) : (
                    <div className="d-flex flex-column gap-4 fade-in delay-1" style={{ maxWidth: '1000px', margin: '0 auto' }}>
                        <div className="d-flex justify-between align-center">
                            <h2>Analysis Results for <span style={{ color: 'var(--accent-primary)' }}>{file.name}</span></h2>
                            <button className="btn btn-outline" onClick={resetAnalysis}>Analyze Another</button>
                        </div>

                        <div className="d-flex gap-4" style={{ flexWrap: 'wrap' }}>
                            {/* Left Column: Verdict */}
                            <div className="glass-panel" style={{ flex: '1 1 300px' }}>
                                <h4 style={{ marginBottom: '1.5rem', opacity: 0.8, textTransform: 'uppercase', letterSpacing: '1px' }}>AI Verdict</h4>
                                <div className="text-center" style={{ padding: '2rem 0' }}>
                                    <div style={{
                                        width: '120px', height: '120px', borderRadius: '50%',
                                        background: 'rgba(245, 158, 11, 0.1)', border: '4px solid var(--warning)',
                                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                                        margin: '0 auto 1.5rem auto', boxShadow: '0 0 30px rgba(245, 158, 11, 0.2)'
                                    }}>
                                        <span style={{ fontSize: '1.5rem', fontWeight: '800', color: result.decision === 'FAIR' ? 'var(--success)' : result.decision === 'FLAGGED' ? 'var(--danger)' : 'var(--warning)' }}>
                                            {result.decision.replace(/_/g, ' ')}
                                        </span>
                                    </div>
                                    <h3 style={{ marginBottom: '0.5rem' }}>{result.summary}</h3>
                                    <p>Confidence Score: <span style={{ color: 'var(--text-primary)', fontWeight: 'bold' }}>{result.confidence}%</span></p>
                                </div>
                            </div>

                            {/* Right Column: Key Metrics & Extraction */}
                            <div className="glass-panel d-flex flex-column gap-4" style={{ flex: '2 1 500px' }}>
                                <div>
                                    <h4 style={{ marginBottom: '1rem', opacity: 0.8, textTransform: 'uppercase', letterSpacing: '1px' }}>Extracted Claim Data</h4>
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                                        <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '12px' }}>
                                            <p style={{ fontSize: '0.85rem', marginBottom: '0.2rem' }}>Vehicle Class</p>
                                            <strong style={{ fontSize: '1.2rem', color: 'var(--accent-primary)' }}>{result.extractedData.vehicleClass}</strong>
                                        </div>
                                        <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '12px' }}>
                                            <p style={{ fontSize: '0.85rem', marginBottom: '0.2rem' }}>Duration</p>
                                            <strong style={{ fontSize: '1.2rem' }}>{result.extractedData.startDate} to {result.extractedData.endDate}</strong>
                                        </div>
                                        <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', padding: '1rem', borderRadius: '12px' }}>
                                            <p style={{ fontSize: '0.85rem', marginBottom: '0.2rem' }}>Claimed Daily Rate</p>
                                            <strong style={{ fontSize: '1.5rem', color: 'var(--danger)' }}>£{result.extractedData.dailyRate}</strong>
                                        </div>
                                        <div style={{ background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.2)', padding: '1rem', borderRadius: '12px' }}>
                                            <p style={{ fontSize: '0.85rem', marginBottom: '0.2rem' }}>RAG Market Average</p>
                                            <strong style={{ fontSize: '1.5rem', color: 'var(--success)' }}>£{result.marketRates.average}</strong>
                                        </div>
                                    </div>
                                </div>

                                <div>
                                    <h4 style={{ marginBottom: '1rem', opacity: 0.8, textTransform: 'uppercase', letterSpacing: '1px' }}>Model Explanation (SHAP)</h4>
                                    <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                                        {result.explanation.map((exp, i) => (
                                            <li key={i} style={{
                                                background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '12px',
                                                marginBottom: '0.5rem', borderLeft: '4px solid var(--accent-primary)',
                                                display: 'flex', gap: '1rem', alignItems: 'flex-start'
                                            }}>
                                                <span style={{ color: 'var(--accent-primary)', fontWeight: 'bold' }}>0{i + 1}</span>
                                                <span>{exp}</span>
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </main>

            <style dangerouslySetInnerHTML={{
                __html: `
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}} />
        </div>
    );
}

export default App;
