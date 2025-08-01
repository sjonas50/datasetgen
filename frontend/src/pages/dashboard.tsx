import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';

interface Dataset {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
}

interface Pipeline {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
}

export default function Dashboard() {
  const router = useRouter();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [pipelines, setPipelines] = useState<Pipeline[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'datasets' | 'pipelines'>('datasets');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [createType, setCreateType] = useState<'dataset' | 'pipeline'>('dataset');
  const [newItemName, setNewItemName] = useState('');
  const [newItemDescription, setNewItemDescription] = useState('');

  useEffect(() => {
    fetchData();
  }, []);

  const getAuthHeaders = () => {
    const token = localStorage.getItem('auth_token');
    if (!token) {
      router.push('/login');
      return null;
    }
    return {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    };
  };

  const fetchData = async () => {
    const headers = getAuthHeaders();
    if (!headers) return;

    try {
      const [datasetsRes, pipelinesRes] = await Promise.all([
        fetch('http://localhost:8000/api/v1/datasets', { headers }),
        fetch('http://localhost:8000/api/v1/pipelines', { headers }),
      ]);

      if (!datasetsRes.ok || !pipelinesRes.ok) {
        if (datasetsRes.status === 401 || pipelinesRes.status === 401) {
          router.push('/login');
          return;
        }
        throw new Error('Failed to fetch data');
      }

      const datasetsData = await datasetsRes.json();
      const pipelinesData = await pipelinesRes.json();

      setDatasets(datasetsData);
      setPipelines(pipelinesData);
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async () => {
    const headers = getAuthHeaders();
    if (!headers) return;

    const endpoint = createType === 'dataset' 
      ? 'http://localhost:8000/api/v1/datasets'
      : 'http://localhost:8000/api/v1/pipelines';

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          name: newItemName,
          description: newItemDescription || null,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to create');
      }

      // Refresh data
      await fetchData();
      
      // Reset form
      setShowCreateModal(false);
      setNewItemName('');
      setNewItemDescription('');
    } catch (error) {
      console.error('Error creating item:', error);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('auth_token');
    router.push('/login');
  };

  if (loading) {
    return (
      <div style={styles.container}>
        <div style={styles.loading}>Loading...</div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.logo}>DatasetGen</h1>
        <nav style={styles.nav}>
          <a href="/dashboard" style={{ ...styles.navLink, ...styles.activeNav }}>Dashboard</a>
          <a href="/datasets" style={styles.navLink}>Datasets</a>
          <a href="/pipelines" style={styles.navLink}>Pipelines</a>
          <button onClick={handleLogout} style={styles.logoutButton}>
            Logout
          </button>
        </nav>
      </header>

      <main style={styles.main}>
        <div style={styles.welcomeSection}>
          <h2>Welcome to DatasetGen</h2>
          <p style={styles.subtitle}>
            Transform your raw data into high-quality ML training datasets with AI-powered processing
          </p>
        </div>

        <div style={styles.statsGrid}>
          <div style={styles.statCard}>
            <h3 style={styles.statNumber}>{datasets.length}</h3>
            <p style={styles.statLabel}>Datasets</p>
          </div>
          <div style={styles.statCard}>
            <h3 style={styles.statNumber}>{pipelines.length}</h3>
            <p style={styles.statLabel}>Pipelines</p>
          </div>
          <div style={styles.statCard}>
            <h3 style={styles.statNumber}>0</h3>
            <p style={styles.statLabel}>Executions</p>
          </div>
          <div style={styles.statCard}>
            <h3 style={styles.statNumber}>--</h3>
            <p style={styles.statLabel}>Quality Score</p>
          </div>
        </div>

        <div style={styles.quickActions}>
          <h3>Quick Actions</h3>
          <div style={styles.actionButtons}>
            <button 
              onClick={() => router.push('/datasets')} 
              style={styles.actionButton}
            >
              📁 Create Dataset
            </button>
            <button 
              onClick={() => router.push('/pipelines')} 
              style={styles.actionButton}
            >
              🔄 Build Pipeline
            </button>
            <button 
              onClick={() => router.push('/datasets')} 
              style={styles.actionButton}
            >
              📊 View Reports
            </button>
          </div>
        </div>

        <div style={styles.recentSection}>
          <h3>Recent Activity</h3>
          {datasets.length === 0 && pipelines.length === 0 ? (
            <div style={styles.emptyState}>
              <p>No activity yet. Start by creating a dataset!</p>
              <button 
                onClick={() => router.push('/datasets')} 
                style={styles.primaryButton}
              >
                Get Started
              </button>
            </div>
          ) : (
            <div style={styles.activityList}>
              {datasets.slice(0, 3).map(dataset => (
                <div key={dataset.id} style={styles.activityItem}>
                  <div style={styles.activityIcon}>📁</div>
                  <div style={styles.activityContent}>
                    <p style={styles.activityTitle}>Dataset created: {dataset.name}</p>
                    <p style={styles.activityDate}>
                      {new Date(dataset.created_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              ))}
              {pipelines.slice(0, 3).map(pipeline => (
                <div key={pipeline.id} style={styles.activityItem}>
                  <div style={styles.activityIcon}>🔄</div>
                  <div style={styles.activityContent}>
                    <p style={styles.activityTitle}>Pipeline created: {pipeline.name}</p>
                    <p style={styles.activityDate}>
                      {new Date(pipeline.created_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

const styles = {
  container: {
    minHeight: '100vh',
    background: '#f5f5f5',
    fontFamily: 'Arial, sans-serif',
  },
  header: {
    background: 'white',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
    padding: '20px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  logo: {
    margin: 0,
    color: '#2563eb',
  },
  nav: {
    display: 'flex',
    gap: '30px',
    alignItems: 'center',
  },
  navLink: {
    color: '#666',
    textDecoration: 'none',
    fontSize: '16px',
  },
  activeNav: {
    color: '#2563eb',
    fontWeight: 'bold',
  },
  logoutButton: {
    padding: '8px 16px',
    background: 'transparent',
    color: '#666',
    border: '1px solid #ddd',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  main: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '20px',
  },
  welcomeSection: {
    textAlign: 'center' as const,
    marginBottom: '40px',
  },
  subtitle: {
    color: '#666',
    fontSize: '18px',
    marginTop: '10px',
  },
  statsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
    gap: '20px',
    marginBottom: '40px',
  },
  statCard: {
    background: 'white',
    padding: '30px',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
    textAlign: 'center' as const,
  },
  statNumber: {
    fontSize: '36px',
    fontWeight: 'bold',
    color: '#2563eb',
    margin: '0 0 10px 0',
  },
  statLabel: {
    color: '#666',
    fontSize: '16px',
    margin: 0,
  },
  quickActions: {
    marginBottom: '40px',
  },
  actionButtons: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '15px',
    marginTop: '20px',
  },
  actionButton: {
    padding: '20px',
    background: 'white',
    border: '2px solid #e5e5e5',
    borderRadius: '8px',
    fontSize: '16px',
    cursor: 'pointer',
    transition: 'all 0.3s',
    '&:hover': {
      borderColor: '#2563eb',
      transform: 'translateY(-2px)',
    },
  },
  recentSection: {
    background: 'white',
    padding: '30px',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
  },
  emptyState: {
    textAlign: 'center' as const,
    padding: '40px',
    color: '#666',
  },
  primaryButton: {
    padding: '12px 24px',
    background: '#2563eb',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '16px',
    marginTop: '20px',
  },
  activityList: {
    marginTop: '20px',
  },
  activityItem: {
    display: 'flex',
    gap: '15px',
    padding: '15px',
    borderBottom: '1px solid #f0f0f0',
    alignItems: 'center',
  },
  activityIcon: {
    fontSize: '24px',
  },
  activityContent: {
    flex: 1,
  },
  activityTitle: {
    margin: '0 0 5px 0',
    color: '#333',
  },
  activityDate: {
    margin: 0,
    fontSize: '12px',
    color: '#999',
  },
  loading: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    height: '100vh',
    fontSize: '18px',
    color: '#666',
  },
};