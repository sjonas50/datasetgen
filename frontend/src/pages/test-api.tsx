import { useState } from 'react';

export default function TestAPI() {
  const [results, setResults] = useState<string[]>([]);
  
  const addResult = (message: string) => {
    setResults(prev => [...prev, `${new Date().toLocaleTimeString()}: ${message}`]);
  };

  const testRegister = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/v1/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: `test${Date.now()}@example.com`,
          username: `testuser${Date.now()}`,
          password: 'testpass123',
        }),
      });
      const data = await response.json();
      addResult(`Register: ${response.ok ? 'Success' : 'Failed'} - ${JSON.stringify(data)}`);
    } catch (error) {
      addResult(`Register Error: ${error}`);
    }
  };

  const testLogin = async () => {
    try {
      const formData = new FormData();
      formData.append('username', 'testuser');
      formData.append('password', 'testpass123');
      
      const response = await fetch('http://localhost:8000/api/v1/auth/login', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        localStorage.setItem('auth_token', data.access_token);
        addResult(`Login: Success - Token saved`);
      } else {
        addResult(`Login: Failed - ${JSON.stringify(data)}`);
      }
    } catch (error) {
      addResult(`Login Error: ${error}`);
    }
  };

  const testDatasets = async () => {
    try {
      const token = localStorage.getItem('auth_token');
      const response = await fetch('http://localhost:8000/api/v1/datasets', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      const data = await response.json();
      addResult(`Datasets: ${response.ok ? `Success - ${data.length} datasets` : 'Failed'}`);
    } catch (error) {
      addResult(`Datasets Error: ${error}`);
    }
  };

  const clearResults = () => {
    setResults([]);
  };

  return (
    <div style={styles.container}>
      <h1>API Test Page</h1>
      
      <div style={styles.buttons}>
        <button onClick={testRegister} style={styles.button}>
          Test Register
        </button>
        <button onClick={testLogin} style={styles.button}>
          Test Login
        </button>
        <button onClick={testDatasets} style={styles.button}>
          Test Datasets
        </button>
        <button onClick={clearResults} style={styles.clearButton}>
          Clear Results
        </button>
      </div>

      <div style={styles.results}>
        <h2>Results:</h2>
        {results.length === 0 ? (
          <p>No results yet. Click a button to test the API.</p>
        ) : (
          <ul>
            {results.map((result, index) => (
              <li key={index} style={styles.result}>{result}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

const styles = {
  container: {
    padding: '20px',
    maxWidth: '800px',
    margin: '0 auto',
    fontFamily: 'Arial, sans-serif',
  },
  buttons: {
    display: 'flex',
    gap: '10px',
    marginBottom: '20px',
  },
  button: {
    padding: '10px 20px',
    background: '#2563eb',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '16px',
  },
  clearButton: {
    padding: '10px 20px',
    background: '#dc2626',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '16px',
  },
  results: {
    background: '#f5f5f5',
    padding: '20px',
    borderRadius: '8px',
  },
  result: {
    marginBottom: '10px',
    wordBreak: 'break-all' as const,
  },
};