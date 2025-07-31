import { api } from './api';

export interface PipelineMetrics {
  pipeline_id: string;
  pipeline_name: string;
  execution_id: string;
  start_time: string;
  end_time?: string;
  status: string;
  total_steps: number;
  completed_steps: number;
  failed_steps: number;
  rows_processed: number;
  execution_time_seconds: number;
  llm_tokens_used: number;
  llm_cost_usd: number;
  compute_cost_usd: number;
  total_cost_usd: number;
  error_message?: string;
}

export interface ActivePipeline extends PipelineMetrics {
  current_duration_seconds: number;
}

export interface CostSummary {
  total_cost: number;
  llm_cost: number;
  compute_cost: number;
  storage_cost: number;
  by_provider: Record<string, number>;
  by_pipeline: Record<string, number>;
  timeline: Array<{ date: string; cost: number }>;
}

export interface PerformanceStats {
  total_executions: number;
  success_rate: number;
  avg_execution_time: number;
  avg_rows_processed: number;
  avg_cost: number;
  total_cost: number;
  total_rows: number;
}

export interface DashboardData {
  summary: {
    active_pipelines: number;
    total_executions_period: number;
    success_rate: number;
    total_cost_period: number;
    avg_execution_time: number;
    total_rows_processed: number;
  };
  active_pipelines: ActivePipeline[];
  recent_failures: PipelineMetrics[];
  cost_breakdown: {
    llm: number;
    compute: number;
    storage: number;
    by_pipeline: Record<string, number>;
    by_provider: Record<string, number>;
  };
  execution_trend: Array<{
    date: string;
    total: number;
    successful: number;
    failed: number;
  }>;
  cost_trend: Array<{ date: string; cost: number }>;
  period_days: number;
}

class MonitoringService {
  async getPipelineMetrics(
    params?: {
      execution_id?: string;
      pipeline_id?: string;
      start_date?: string;
      end_date?: string;
      limit?: number;
    }
  ): Promise<{ total: number; metrics: PipelineMetrics[] }> {
    const response = await api.get('/monitoring/metrics/pipelines', { params });
    return response.data;
  }

  async getActivePipelines(): Promise<{ count: number; pipelines: ActivePipeline[] }> {
    const response = await api.get('/monitoring/metrics/active-pipelines');
    return response.data;
  }

  async getCostSummary(
    start_date: string,
    end_date: string,
    group_by: 'day' | 'week' | 'month' = 'day'
  ): Promise<CostSummary> {
    const response = await api.get('/monitoring/metrics/costs', {
      params: { start_date, end_date, group_by },
    });
    return response.data;
  }

  async getPerformanceStats(
    pipeline_id?: string,
    days: number = 7
  ): Promise<PerformanceStats> {
    const response = await api.get('/monitoring/metrics/performance', {
      params: { pipeline_id, days },
    });
    return response.data;
  }

  async getLLMUsage(
    start_date: string,
    end_date: string,
    provider?: string,
    model?: string
  ): Promise<{
    total_llm_cost: number;
    by_provider: Record<string, number>;
    timeline: Array<{ date: string; cost: number }>;
  }> {
    const response = await api.get('/monitoring/metrics/llm-usage', {
      params: { start_date, end_date, provider, model },
    });
    return response.data;
  }

  async getDashboard(days: number = 7): Promise<DashboardData> {
    const response = await api.get('/monitoring/metrics/dashboard', {
      params: { days },
    });
    return response.data;
  }

  async testAlert(
    alert_type: 'cost_threshold' | 'error_rate' | 'execution_time',
    threshold: number
  ): Promise<{ status: string; message: string }> {
    const response = await api.post('/monitoring/metrics/alert-test', null, {
      params: { alert_type, threshold },
    });
    return response.data;
  }
}

export const monitoringService = new MonitoringService();