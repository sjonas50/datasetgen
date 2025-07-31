import axios, { AxiosInstance } from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Create axios instance
const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Dataset Service
export const datasetService = {
  list: (params?: { skip?: number; limit?: number; search?: string }) =>
    api.get('/api/v1/datasets', { params }),
  
  get: (id: string) => api.get(`/api/v1/datasets/${id}`),
  
  create: (data: any) => api.post('/api/v1/datasets', data),
  
  update: (id: string, data: any) => api.patch(`/api/v1/datasets/${id}`, data),
  
  delete: (id: string) => api.delete(`/api/v1/datasets/${id}`),
  
  upload: (id: string, file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post(`/api/v1/datasets/${id}/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
};

// Pipeline Service
export const pipelineService = {
  list: (params?: { skip?: number; limit?: number }) =>
    api.get('/api/v1/pipelines', { params }),
  
  get: (id: string) => api.get(`/api/v1/pipelines/${id}`),
  
  create: (data: any) => api.post('/api/v1/pipelines', data),
  
  update: (id: string, data: any) => api.patch(`/api/v1/pipelines/${id}`, data),
  
  delete: (id: string) => api.delete(`/api/v1/pipelines/${id}`),
  
  execute: (id: string, data: any) => api.post(`/api/v1/pipelines/${id}/execute`, data),
  
  getExecutions: (id: string) => api.get(`/api/v1/pipelines/${id}/executions`),
};

// Connector Service
export const connectorService = {
  list: () => api.get('/api/v1/connectors'),
  
  get: (type: string) => api.get(`/api/v1/connectors/${type}`),
  
  test: (type: string, config: any) => api.post(`/api/v1/connectors/${type}/test`, config),
  
  getSchema: (type: string, config: any) => api.post(`/api/v1/connectors/${type}/schema`, config),
};

// Auth Service
export const authService = {
  login: (username: string, password: string) => {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    return api.post('/api/v1/auth/token', formData);
  },
  
  register: (data: { email: string; username: string; password: string }) =>
    api.post('/api/v1/auth/register', data),
  
  getCurrentUser: () => api.get('/api/v1/auth/me'),
};

export default api;