const API_BASE_URL = 'http://localhost:8000/api/v1';

export class APIClient {
  private static getAuthHeaders(): HeadersInit {
    const token = localStorage.getItem('auth_token');
    return {
      'Content-Type': 'application/json',
      ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
    };
  }

  static async login(username: string, password: string) {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    const response = await fetch(`${API_BASE_URL}/auth/login`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Invalid credentials');
    }

    const data = await response.json();
    localStorage.setItem('auth_token', data.access_token);
    return data;
  }

  static async register(email: string, username: string, password: string) {
    const response = await fetch(`${API_BASE_URL}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, username, password }),
    });

    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.detail || 'Registration failed');
    }

    return await response.json();
  }

  static async getDatasets() {
    const response = await fetch(`${API_BASE_URL}/datasets`, {
      headers: this.getAuthHeaders(),
    });

    if (!response.ok) {
      if (response.status === 401) {
        throw new Error('Unauthorized');
      }
      throw new Error('Failed to fetch datasets');
    }

    return await response.json();
  }

  static async createDataset(name: string, description?: string, schema?: any) {
    const response = await fetch(`${API_BASE_URL}/datasets`, {
      method: 'POST',
      headers: this.getAuthHeaders(),
      body: JSON.stringify({ name, description, schema }),
    });

    if (!response.ok) {
      throw new Error('Failed to create dataset');
    }

    return await response.json();
  }

  static async getPipelines() {
    const response = await fetch(`${API_BASE_URL}/pipelines`, {
      headers: this.getAuthHeaders(),
    });

    if (!response.ok) {
      if (response.status === 401) {
        throw new Error('Unauthorized');
      }
      throw new Error('Failed to fetch pipelines');
    }

    return await response.json();
  }

  static async createPipeline(name: string, description?: string, config?: any) {
    const response = await fetch(`${API_BASE_URL}/pipelines`, {
      method: 'POST',
      headers: this.getAuthHeaders(),
      body: JSON.stringify({ name, description, config }),
    });

    if (!response.ok) {
      throw new Error('Failed to create pipeline');
    }

    return await response.json();
  }

  static logout() {
    localStorage.removeItem('auth_token');
  }

  static isAuthenticated(): boolean {
    return !!localStorage.getItem('auth_token');
  }
}