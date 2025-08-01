import React, { useCallback, useMemo, useState, useEffect } from 'react';
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Edge,
  Node,
  MarkerType,
  NodeProps,
  Handle,
  Position,
  ReactFlowProvider,
  useReactFlow,
  Panel,
} from 'reactflow';
import { Card, Button, Space, Drawer, Form, Input, Select, InputNumber, Switch, message, Tag, Tooltip, Modal, List, Badge } from 'antd';
import {
  PlusOutlined,
  SaveOutlined,
  PlayCircleOutlined,
  DatabaseOutlined,
  SafetyOutlined,
  ClearOutlined,
  LineChartOutlined,
  SwapOutlined,
  CheckCircleOutlined,
  DeleteOutlined,
  SettingOutlined,
  QuestionCircleOutlined,
} from '@ant-design/icons';
import 'reactflow/dist/style.css';
import api from '@/services/api';

const { Option } = Select;

// Processing step configurations
const PROCESSING_STEPS = [
  {
    type: 'quality_validation',
    label: 'Quality Validation',
    icon: <CheckCircleOutlined />,
    color: '#1890ff',
    description: 'Validate data quality and identify issues',
    config: {
      min_quality_score: { type: 'number', label: 'Minimum Quality Score', default: 70 },
      fail_on_critical: { type: 'boolean', label: 'Fail on Critical Issues', default: true },
    },
  },
  {
    type: 'pii_detection',
    label: 'PII Detection',
    icon: <SafetyOutlined />,
    color: '#f5222d',
    description: 'Detect and handle personal information',
    config: {
      action: { type: 'select', label: 'Action', options: ['mask', 'remove', 'flag'], default: 'mask' },
      patterns: { type: 'multiselect', label: 'PII Types', options: ['ssn', 'email', 'phone', 'credit_card', 'ip_address'], default: ['ssn', 'email', 'phone'] },
    },
  },
  {
    type: 'data_cleaning',
    label: 'Data Cleaning',
    icon: <ClearOutlined />,
    color: '#52c41a',
    description: 'Clean and standardize data',
    config: {
      remove_duplicates: { type: 'boolean', label: 'Remove Duplicates', default: true },
      trim_whitespace: { type: 'boolean', label: 'Trim Whitespace', default: true },
      handle_missing: { type: 'select', label: 'Handle Missing Values', options: ['drop', 'fill', 'keep'], default: 'drop' },
      standardize_casing: { type: 'boolean', label: 'Standardize Casing', default: true },
      casing_type: { type: 'select', label: 'Casing Type', options: ['lower', 'upper', 'title'], default: 'lower', depends_on: 'standardize_casing' },
    },
  },
  {
    type: 'outlier_detection',
    label: 'Outlier Detection',
    icon: <LineChartOutlined />,
    color: '#fa8c16',
    description: 'Detect and handle outliers',
    config: {
      method: { type: 'select', label: 'Detection Method', options: ['zscore', 'iqr'], default: 'zscore' },
      threshold: { type: 'number', label: 'Threshold', default: 3 },
      action: { type: 'select', label: 'Action', options: ['flag', 'remove', 'cap'], default: 'flag' },
    },
  },
  {
    type: 'data_transformation',
    label: 'AI Transformation',
    icon: <SwapOutlined />,
    color: '#722ed1',
    description: 'Transform data using AI',
    config: {
      prompt_template: { type: 'textarea', label: 'Transformation Prompt', default: 'Transform the data to...', required: true },
      target_schema: { type: 'json', label: 'Target Schema (optional)', default: null },
    },
  },
  {
    type: 'schema_validation',
    label: 'Schema Validation',
    icon: <DatabaseOutlined />,
    color: '#13c2c2',
    description: 'Validate data against schema',
    config: {
      strict_mode: { type: 'boolean', label: 'Strict Mode', default: true },
      allow_extra_columns: { type: 'boolean', label: 'Allow Extra Columns', default: false },
      schema: { type: 'json', label: 'Schema Definition', default: { required_columns: [], column_types: {} }, required: true },
    },
  },
];

// Custom node components
function SourceNode({ data }: NodeProps) {
  return (
    <Card
      size="small"
      title={
        <Space>
          <DatabaseOutlined />
          <span>Dataset: {data.label}</span>
        </Space>
      }
      style={{ minWidth: 200 }}
      bodyStyle={{ padding: 8 }}
    >
      <div style={{ fontSize: 12, color: '#666' }}>
        {data.rows} rows × {data.columns} columns
      </div>
      <Handle type="source" position={Position.Right} style={{ background: '#1890ff' }} />
    </Card>
  );
}

function ProcessingNode({ data }: NodeProps) {
  const stepConfig = PROCESSING_STEPS.find(s => s.type === data.type);
  
  return (
    <Card
      size="small"
      title={
        <Space>
          {stepConfig?.icon}
          <span>{data.label}</span>
          <Tooltip title="Configure">
            <SettingOutlined 
              style={{ cursor: 'pointer', fontSize: 12 }} 
              onClick={() => data.onConfigure?.(data.id)}
            />
          </Tooltip>
        </Space>
      }
      style={{ 
        minWidth: 200,
        borderColor: stepConfig?.color,
        borderWidth: 2,
      }}
      bodyStyle={{ padding: 8 }}
    >
      <Handle type="target" position={Position.Left} style={{ background: '#666' }} />
      <div style={{ fontSize: 11, color: '#666' }}>
        {stepConfig?.description}
      </div>
      {data.configured && (
        <Tag color="success" style={{ marginTop: 4 }}>Configured</Tag>
      )}
      <Handle type="source" position={Position.Right} style={{ background: stepConfig?.color }} />
    </Card>
  );
}

function OutputNode({ data }: NodeProps) {
  return (
    <Card
      size="small"
      title={
        <Space>
          <CheckCircleOutlined style={{ color: '#52c41a' }} />
          <span>Output Dataset</span>
        </Space>
      }
      style={{ minWidth: 200 }}
      bodyStyle={{ padding: 8 }}
    >
      <Handle type="target" position={Position.Left} style={{ background: '#666' }} />
      <div style={{ fontSize: 12, color: '#666' }}>Processed data output</div>
    </Card>
  );
}

// Node types (defined after components)
const nodeTypes = {
  sourceNode: SourceNode,
  qualityValidation: ProcessingNode,
  piiDetection: ProcessingNode,
  dataCleaning: ProcessingNode,
  outlierDetection: ProcessingNode,
  dataTransformation: ProcessingNode,
  schemaValidation: ProcessingNode,
  outputNode: OutputNode,
};

// Main component
interface PipelineBuilderProps {
  datasetId: string;
  datasetName: string;
  datasetInfo: {
    rows: number;
    columns: number;
  };
  onSave?: (pipeline: any) => void;
  existingPipeline?: any;
}

function PipelineBuilderContent({ datasetId, datasetName, datasetInfo, onSave, existingPipeline }: PipelineBuilderProps) {
  const reactFlowInstance = useReactFlow();
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [nodeConfigs, setNodeConfigs] = useState<Record<string, any>>({});
  const [pipelineName, setPipelineName] = useState(existingPipeline?.name || '');
  const [pipelineDescription, setPipelineDescription] = useState(existingPipeline?.description || '');
  const [executing, setExecuting] = useState(false);
  const [executionModal, setExecutionModal] = useState(false);
  const [executions, setExecutions] = useState<any[]>([]);
  const [form] = Form.useForm();

  // Initialize with source and output nodes
  useEffect(() => {
    if (existingPipeline) {
      // Load existing pipeline
      loadPipelineFromConfig(existingPipeline);
    } else {
      // Create new pipeline
      const sourceNode: Node = {
        id: 'source',
        type: 'sourceNode',
        data: { 
          label: datasetName,
          rows: datasetInfo.rows,
          columns: datasetInfo.columns,
        },
        position: { x: 50, y: 200 },
      };

      const outputNode: Node = {
        id: 'output',
        type: 'outputNode',
        data: { label: 'Output' },
        position: { x: 800, y: 200 },
      };

      setNodes([sourceNode, outputNode]);
    }
  }, [datasetName, datasetInfo, existingPipeline]);

  const loadPipelineFromConfig = (pipeline: any) => {
    // Convert pipeline steps to nodes and edges
    const newNodes: Node[] = [];
    const newEdges: Edge[] = [];
    const configs: Record<string, any> = {};

    // Add source node
    newNodes.push({
      id: 'source',
      type: 'sourceNode',
      data: { 
        label: datasetName,
        rows: datasetInfo.rows,
        columns: datasetInfo.columns,
      },
      position: { x: 50, y: 200 },
    });

    // Add processing nodes
    let previousNodeId = 'source';
    pipeline.steps.forEach((step: any, index: number) => {
      const nodeId = `node_${index}`;
      const stepConfig = PROCESSING_STEPS.find(s => s.type === step.type);
      
      newNodes.push({
        id: nodeId,
        type: step.type,
        data: {
          label: stepConfig?.label || step.type,
          type: step.type,
          configured: true,
          onConfigure: handleNodeConfigure,
        },
        position: { x: 200 + index * 180, y: 200 },
      });

      newEdges.push({
        id: `${previousNodeId}-${nodeId}`,
        source: previousNodeId,
        target: nodeId,
        type: 'smoothstep',
      });

      configs[nodeId] = step.config;
      previousNodeId = nodeId;
    });

    // Add output node
    newNodes.push({
      id: 'output',
      type: 'outputNode',
      data: { label: 'Output' },
      position: { x: 200 + pipeline.steps.length * 180, y: 200 },
    });

    newEdges.push({
      id: `${previousNodeId}-output`,
      source: previousNodeId,
      target: 'output',
      type: 'smoothstep',
    });

    setNodes(newNodes);
    setEdges(newEdges);
    setNodeConfigs(configs);
  };

  const onConnect = useCallback(
    (params: Connection) => {
      setEdges((eds) => addEdge({ ...params, type: 'smoothstep' }, eds));
    },
    [setEdges]
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const type = event.dataTransfer.getData('application/reactflow');
      const stepConfig = PROCESSING_STEPS.find(s => s.type === type);

      if (!type || !stepConfig) {
        return;
      }

      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const newNode: Node = {
        id: `${type}_${Date.now()}`,
        type,
        position,
        data: { 
          label: stepConfig.label,
          type,
          configured: false,
          onConfigure: handleNodeConfigure,
        },
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance]
  );

  const handleNodeConfigure = (nodeId: string) => {
    setSelectedNode(nodeId);
    const node = nodes.find(n => n.id === nodeId);
    if (node) {
      const existingConfig = nodeConfigs[nodeId] || {};
      const stepConfig = PROCESSING_STEPS.find(s => s.type === node.data.type);
      
      // Set form values
      const formValues: any = {};
      if (stepConfig) {
        Object.entries(stepConfig.config).forEach(([key, config]: [string, any]) => {
          formValues[key] = existingConfig[key] ?? config.default;
        });
      }
      form.setFieldsValue(formValues);
      setDrawerVisible(true);
    }
  };

  const handleConfigSave = () => {
    form.validateFields().then(values => {
      if (selectedNode) {
        setNodeConfigs(prev => ({ ...prev, [selectedNode]: values }));
        setNodes(nds => 
          nds.map(node => 
            node.id === selectedNode 
              ? { ...node, data: { ...node.data, configured: true } }
              : node
          )
        );
        setDrawerVisible(false);
        message.success('Configuration saved');
      }
    });
  };

  const handleDeleteNode = () => {
    if (selectedNode && selectedNode !== 'source' && selectedNode !== 'output') {
      setNodes(nds => nds.filter(n => n.id !== selectedNode));
      setEdges(eds => eds.filter(e => e.source !== selectedNode && e.target !== selectedNode));
      setNodeConfigs(prev => {
        const newConfigs = { ...prev };
        delete newConfigs[selectedNode];
        return newConfigs;
      });
      setDrawerVisible(false);
      message.success('Node deleted');
    }
  };

  const buildPipelineConfig = () => {
    // Get nodes in execution order (topological sort)
    const nodeMap = new Map(nodes.map(n => [n.id, n]));
    const adjList = new Map<string, string[]>();
    
    // Build adjacency list
    edges.forEach(edge => {
      if (!adjList.has(edge.source)) {
        adjList.set(edge.source, []);
      }
      adjList.get(edge.source)!.push(edge.target);
    });

    // Find path from source to output
    const visited = new Set<string>();
    const path: string[] = [];
    
    const dfs = (nodeId: string) => {
      if (nodeId === 'output') return true;
      if (visited.has(nodeId)) return false;
      
      visited.add(nodeId);
      const neighbors = adjList.get(nodeId) || [];
      
      for (const neighbor of neighbors) {
        if (dfs(neighbor)) {
          if (nodeId !== 'source') {
            path.unshift(nodeId);
          }
          return true;
        }
      }
      
      return false;
    };

    dfs('source');

    // Build steps from path
    const steps = path.map(nodeId => {
      const node = nodeMap.get(nodeId)!;
      const config = nodeConfigs[nodeId] || {};
      return {
        type: node.data.type,
        config,
      };
    });

    return {
      name: pipelineName || 'Untitled Pipeline',
      description: pipelineDescription,
      dataset_id: datasetId,
      steps,
    };
  };

  const handleSavePipeline = async () => {
    if (!pipelineName) {
      message.error('Please enter a pipeline name');
      return;
    }

    // Validate all nodes are configured
    const unconfiguredNodes = nodes.filter(
      n => n.id !== 'source' && n.id !== 'output' && !n.data.configured
    );
    
    if (unconfiguredNodes.length > 0) {
      message.error('Please configure all processing nodes before saving');
      return;
    }

    try {
      const pipelineConfig = buildPipelineConfig();
      
      if (existingPipeline?.id) {
        // Update existing pipeline
        await api.put(`/pipelines/${existingPipeline.id}`, pipelineConfig);
        message.success('Pipeline updated successfully');
      } else {
        // Create new pipeline
        await api.post('/api/v1/pipelines', pipelineConfig);
        message.success('Pipeline saved successfully');
      }
      
      onSave?.(pipelineConfig);
    } catch (error) {
      message.error('Failed to save pipeline');
      console.error(error);
    }
  };

  const handleExecutePipeline = async () => {
    try {
      const pipelineConfig = buildPipelineConfig();
      
      // Save pipeline first if not saved
      let pipelineId = existingPipeline?.id;
      if (!pipelineId) {
        const response = await api.post('/api/v1/pipelines', pipelineConfig);
        pipelineId = response.data.id;
      }

      setExecuting(true);
      const execution = await api.post(`/api/v1/pipelines/${pipelineId}/execute`);
      message.success('Pipeline execution started');
      
      // Show execution status
      setExecutionModal(true);
      loadExecutions(pipelineId);
      
      // Poll for execution status
      const pollInterval = setInterval(async () => {
        try {
          const response = await api.get(`/api/v1/pipelines/${pipelineId}/executions`);
          const latestExecution = response.data[0];
          
          if (latestExecution?.status === 'completed' || latestExecution?.status === 'failed') {
            clearInterval(pollInterval);
            setExecuting(false);
            
            if (latestExecution.status === 'completed') {
              message.success('Pipeline execution completed');
            } else {
              message.error('Pipeline execution failed');
            }
          }
          
          setExecutions(response.data);
        } catch (error) {
          clearInterval(pollInterval);
          setExecuting(false);
        }
      }, 2000);
      
    } catch (error) {
      message.error('Failed to execute pipeline');
      setExecuting(false);
      console.error(error);
    }
  };

  const loadExecutions = async (pipelineId: string) => {
    try {
      const response = await api.get(`/api/v1/pipelines/${pipelineId}/executions`);
      setExecutions(response.data);
    } catch (error) {
      console.error('Failed to load executions:', error);
    }
  };

  const renderConfigField = (key: string, config: any) => {
    switch (config.type) {
      case 'number':
        return (
          <Form.Item
            key={key}
            name={key}
            label={config.label}
            rules={[{ required: config.required }]}
          >
            <InputNumber style={{ width: '100%' }} />
          </Form.Item>
        );
      
      case 'boolean':
        return (
          <Form.Item
            key={key}
            name={key}
            label={config.label}
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>
        );
      
      case 'select':
        return (
          <Form.Item
            key={key}
            name={key}
            label={config.label}
            rules={[{ required: config.required }]}
          >
            <Select>
              {config.options.map((opt: string) => (
                <Option key={opt} value={opt}>{opt}</Option>
              ))}
            </Select>
          </Form.Item>
        );
      
      case 'multiselect':
        return (
          <Form.Item
            key={key}
            name={key}
            label={config.label}
            rules={[{ required: config.required }]}
          >
            <Select mode="multiple">
              {config.options.map((opt: string) => (
                <Option key={opt} value={opt}>{opt}</Option>
              ))}
            </Select>
          </Form.Item>
        );
      
      case 'textarea':
        return (
          <Form.Item
            key={key}
            name={key}
            label={config.label}
            rules={[{ required: config.required }]}
          >
            <Input.TextArea rows={4} />
          </Form.Item>
        );
      
      case 'json':
        return (
          <Form.Item
            key={key}
            name={key}
            label={config.label}
            rules={[
              { required: config.required },
              {
                validator: (_, value) => {
                  if (!value) return Promise.resolve();
                  try {
                    if (typeof value === 'string') {
                      JSON.parse(value);
                    }
                    return Promise.resolve();
                  } catch (e) {
                    return Promise.reject('Invalid JSON format');
                  }
                },
              },
            ]}
          >
            <Input.TextArea 
              rows={6}
              placeholder="Enter valid JSON"
            />
          </Form.Item>
        );
      
      default:
        return (
          <Form.Item
            key={key}
            name={key}
            label={config.label}
            rules={[{ required: config.required }]}
          >
            <Input />
          </Form.Item>
        );
    }
  };

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Card size="small" style={{ marginBottom: 16 }}>
        <Space style={{ width: '100%', justifyContent: 'space-between' }}>
          <Space>
            <Input
              placeholder="Pipeline Name"
              value={pipelineName}
              onChange={(e) => setPipelineName(e.target.value)}
              style={{ width: 200 }}
            />
            <Input
              placeholder="Description (optional)"
              value={pipelineDescription}
              onChange={(e) => setPipelineDescription(e.target.value)}
              style={{ width: 300 }}
            />
          </Space>
          <Space>
            <Button 
              type="primary" 
              icon={<SaveOutlined />}
              onClick={handleSavePipeline}
            >
              Save Pipeline
            </Button>
            <Button 
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={handleExecutePipeline}
              loading={executing}
              disabled={nodes.filter(n => n.id !== 'source' && n.id !== 'output').length === 0}
            >
              Execute Pipeline
            </Button>
          </Space>
        </Space>
      </Card>

      <div style={{ flex: 1, display: 'flex' }}>
        <Card 
          title="Processing Steps" 
          size="small" 
          style={{ width: 250, marginRight: 16 }}
          bodyStyle={{ padding: 8 }}
        >
          <div style={{ fontSize: 12, marginBottom: 8, color: '#666' }}>
            Drag steps to the canvas
          </div>
          {PROCESSING_STEPS.map((step) => (
            <div
              key={step.type}
              draggable
              onDragStart={(event) => {
                event.dataTransfer.setData('application/reactflow', step.type);
                event.dataTransfer.effectAllowed = 'move';
              }}
              style={{
                padding: '8px 12px',
                marginBottom: 8,
                border: `2px solid ${step.color}`,
                borderRadius: 4,
                cursor: 'move',
                backgroundColor: '#fff',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
              }}
            >
              {step.icon}
              <span>{step.label}</span>
              <Tooltip title={step.description}>
                <QuestionCircleOutlined style={{ marginLeft: 'auto', fontSize: 12 }} />
              </Tooltip>
            </div>
          ))}
        </Card>

        <div style={{ flex: 1 }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onDrop={onDrop}
            onDragOver={onDragOver}
            nodeTypes={nodeTypes}
            fitView
            defaultEdgeOptions={{
              type: 'smoothstep',
              markerEnd: {
                type: MarkerType.ArrowClosed,
              },
            }}
          >
            <Background variant="dots" gap={12} size={1} />
            <Controls />
            <MiniMap nodeStrokeWidth={3} />
            <Panel position="top-right">
              <Card size="small">
                <Space direction="vertical" size="small">
                  <div>Nodes: {nodes.length - 2}</div>
                  <div>Connections: {edges.length}</div>
                </Space>
              </Card>
            </Panel>
          </ReactFlow>
        </div>
      </div>

      <Drawer
        title="Configure Step"
        placement="right"
        width={400}
        open={drawerVisible}
        onClose={() => setDrawerVisible(false)}
        footer={
          <Space style={{ float: 'right' }}>
            {selectedNode && selectedNode !== 'source' && selectedNode !== 'output' && (
              <Button danger icon={<DeleteOutlined />} onClick={handleDeleteNode}>
                Delete Node
              </Button>
            )}
            <Button onClick={() => setDrawerVisible(false)}>Cancel</Button>
            <Button type="primary" onClick={handleConfigSave}>Save</Button>
          </Space>
        }
      >
        {selectedNode && (
          <Form form={form} layout="vertical">
            {(() => {
              const node = nodes.find(n => n.id === selectedNode);
              const stepConfig = PROCESSING_STEPS.find(s => s.type === node?.data.type);
              
              if (!stepConfig) return null;
              
              return Object.entries(stepConfig.config).map(([key, config]: [string, any]) => {
                // Check dependencies
                if (config.depends_on) {
                  const dependentValue = form.getFieldValue(config.depends_on);
                  if (!dependentValue) return null;
                }
                
                return renderConfigField(key, config);
              });
            })()}
          </Form>
        )}
      </Drawer>

      <Modal
        title="Pipeline Executions"
        open={executionModal}
        onCancel={() => setExecutionModal(false)}
        footer={[
          <Button key="close" onClick={() => setExecutionModal(false)}>
            Close
          </Button>
        ]}
        width={600}
      >
        <List
          dataSource={executions}
          renderItem={(execution: any) => (
            <List.Item>
              <List.Item.Meta
                title={
                  <Space>
                    <Badge 
                      status={
                        execution.status === 'completed' ? 'success' :
                        execution.status === 'failed' ? 'error' :
                        execution.status === 'running' ? 'processing' :
                        'default'
                      }
                    />
                    <span>Execution {execution.id.slice(0, 8)}</span>
                  </Space>
                }
                description={
                  <div>
                    <div>Started: {new Date(execution.started_at).toLocaleString()}</div>
                    {execution.completed_at && (
                      <div>Completed: {new Date(execution.completed_at).toLocaleString()}</div>
                    )}
                    {execution.error && (
                      <div style={{ color: '#f5222d' }}>Error: {execution.error}</div>
                    )}
                    {execution.metrics && (
                      <div>
                        Input: {execution.metrics.input_records} rows → 
                        Output: {execution.metrics.output_records} rows
                      </div>
                    )}
                  </div>
                }
              />
            </List.Item>
          )}
        />
      </Modal>
    </div>
  );
}

export default function PipelineBuilder(props: PipelineBuilderProps) {
  return (
    <ReactFlowProvider>
      <PipelineBuilderContent {...props} />
    </ReactFlowProvider>
  );
}