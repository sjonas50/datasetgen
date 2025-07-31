import { Node, Edge } from 'reactflow';

export interface StepConfig {
  id: string;
  name: string;
  type: string;
  config: Record<string, any>;
  depends_on?: string[];
  position?: { x: number; y: number };
  llm_config?: {
    provider?: string;
    model?: string;
    temperature?: number;
    use_extended_thinking?: boolean;
    use_vision?: boolean;
  };
}

export interface PipelineConfig {
  id?: string;
  name: string;
  description: string;
  steps: StepConfig[];
  nodes?: Node[];
  edges?: Edge[];
  optimization?: {
    enable_caching?: boolean;
    enable_parallelization?: boolean;
    enable_llm_optimization?: boolean;
  };
  error_handling?: {
    retry_failed_steps?: boolean;
    max_retries?: number;
    save_checkpoints?: boolean;
  };
}

export interface PipelineExecution {
  id: string;
  pipeline_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  started_at?: string;
  completed_at?: string;
  rows_processed?: number;
  error_message?: string;
  result?: any;
}

export interface StepExecutionResult {
  step_name: string;
  status: 'completed' | 'failed' | 'skipped';
  rows_processed?: number;
  execution_time?: number;
  error?: string;
  warnings?: string[];
}