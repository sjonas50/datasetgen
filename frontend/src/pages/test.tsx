export default function Test() {
  return (
    <div style={{ padding: '50px', fontFamily: 'Arial, sans-serif' }}>
      <h1>DatasetGen Platform</h1>
      <p>Welcome to the AI-First Dataset Creation Platform</p>
      
      <div style={{ marginTop: '30px' }}>
        <h2>Quick Links:</h2>
        <ul>
          <li><a href="/login">Login</a></li>
          <li><a href="/datasets">Datasets</a></li>
          <li><a href="/pipelines">Pipelines</a></li>
          <li><a href="/monitoring">Monitoring Dashboard</a></li>
        </ul>
      </div>
      
      <div style={{ marginTop: '30px', padding: '20px', backgroundColor: '#f0f0f0', borderRadius: '8px' }}>
        <h3>Platform Features:</h3>
        <ul>
          <li>✨ Claude Sonnet 4 Integration</li>
          <li>🔄 DocETL-Inspired Pipeline Optimization (30x efficiency)</li>
          <li>📊 Visual Pipeline Builder</li>
          <li>🔍 Automated Quality Validation</li>
          <li>🛡️ PII Detection & Masking</li>
          <li>📈 Real-time Monitoring Dashboard</li>
        </ul>
      </div>
      
      <div style={{ marginTop: '30px' }}>
        <p><strong>API Status:</strong></p>
        <p>Backend: <a href="http://localhost:8000/docs">http://localhost:8000/docs</a></p>
        <p>Frontend: Running on port 3000</p>
      </div>
    </div>
  );
}