import React, { useState, useCallback, useMemo } from 'react';
import ReactFlow, {
  ReactFlowProvider,
  addEdge,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  MiniMap,
  Connection,
  Edge,
  Node,
  NodeTypes,
  MarkerType,
} from 'reactflow';
import { Button, Card, message, Modal, Drawer } from 'antd';
import { SaveOutlined, PlayCircleOutlined, PlusOutlined } from '@ant-design/icons';
import 'reactflow/dist/style.css';

import StepNode from './nodes/StepNode';
import ConnectorNode from './nodes/ConnectorNode';
import TransformNode from './nodes/TransformNode';
import LLMNode from './nodes/LLMNode';
import MultiModalNode from './nodes/MultiModalNode';
import ValidateNode from './nodes/ValidateNode';
import StepLibrary from './StepLibrary';
import StepConfigPanel from './StepConfigPanel';
import { PipelineConfig, StepConfig } from '../../types/pipeline';
import styles from './PipelineBuilder.module.css';

const nodeTypes: NodeTypes = {
  connector: ConnectorNode,
  transform: TransformNode,
  llm_process: LLMNode,
  multimodal: MultiModalNode,
  validate: ValidateNode,
  default: StepNode,
};

interface PipelineBuilderProps {
  pipeline?: PipelineConfig;
  onSave: (pipeline: PipelineConfig) => void;
  onExecute: (pipeline: PipelineConfig) => void;
}

const PipelineBuilder: React.FC<PipelineBuilderProps> = ({
  pipeline,
  onSave,
  onExecute,
}) => {
  const [nodes, setNodes, onNodesChange] = useNodesState(pipeline?.nodes || []);
  const [edges, setEdges, onEdgesChange] = useEdgesState(pipeline?.edges || []);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [showLibrary, setShowLibrary] = useState(false);
  const [showConfig, setShowConfig] = useState(false);

  // Generate pipeline configuration from nodes and edges
  const generatePipelineConfig = useCallback((): PipelineConfig => {
    const steps: StepConfig[] = nodes.map((node) => ({
      id: node.id,
      name: node.data.label || node.id,
      type: node.type || 'default',
      config: node.data.config || {},
      depends_on: edges
        .filter((edge) => edge.target === node.id)
        .map((edge) => edge.source),
      position: node.position,
    }));

    return {
      name: pipeline?.name || 'New Pipeline',
      description: pipeline?.description || '',
      steps,
      nodes,
      edges,
    };
  }, [nodes, edges, pipeline]);

  // Handle node connection
  const onConnect = useCallback(
    (params: Edge | Connection) => {
      const newEdge = {
        ...params,
        markerEnd: {
          type: MarkerType.ArrowClosed,
          width: 20,
          height: 20,
        },
        style: {
          strokeWidth: 2,
          stroke: '#1890ff',
        },
      };
      setEdges((eds) => addEdge(newEdge, eds));
    },
    [setEdges]
  );

  // Handle node selection
  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedNode(node);
    setShowConfig(true);
  }, []);

  // Handle node deletion
  const onNodesDelete = useCallback(
    (deleted: Node[]) => {
      setEdges((eds) =>
        eds.filter(
          (edge) =>
            !deleted.some(
              (node) => node.id === edge.source || node.id === edge.target
            )
        )
      );
    },
    [setEdges]
  );

  // Add new node from library
  const onAddNode = useCallback(
    (stepType: string, stepConfig: any) => {
      const newNode: Node = {
        id: `${stepType}_${Date.now()}`,
        type: stepType,
        position: {
          x: Math.random() * 500 + 100,
          y: Math.random() * 300 + 100,
        },
        data: {
          label: stepConfig.label || stepType,
          config: stepConfig.defaultConfig || {},
          icon: stepConfig.icon,
        },
      };

      setNodes((nds) => nds.concat(newNode));
      setShowLibrary(false);
      message.success(`Added ${stepConfig.label} step`);
    },
    [setNodes]
  );

  // Update node configuration
  const onUpdateNodeConfig = useCallback(
    (nodeId: string, config: any) => {
      setNodes((nds) =>
        nds.map((node) => {
          if (node.id === nodeId) {
            return {
              ...node,
              data: {
                ...node.data,
                config,
              },
            };
          }
          return node;
        })
      );
      message.success('Step configuration updated');
    },
    [setNodes]
  );

  // Save pipeline
  const handleSave = useCallback(() => {
    const config = generatePipelineConfig();
    onSave(config);
    message.success('Pipeline saved successfully');
  }, [generatePipelineConfig, onSave]);

  // Execute pipeline
  const handleExecute = useCallback(() => {
    const config = generatePipelineConfig();
    
    // Validate pipeline
    if (nodes.length === 0) {
      message.error('Pipeline has no steps');
      return;
    }

    // Check for cycles
    // TODO: Implement cycle detection

    onExecute(config);
  }, [generatePipelineConfig, nodes, onExecute]);

  // Edge validation
  const isValidConnection = useCallback(
    (connection: Connection) => {
      // Prevent self-connections
      if (connection.source === connection.target) return false;

      // Prevent duplicate connections
      const existingEdge = edges.find(
        (edge) =>
          edge.source === connection.source && edge.target === connection.target
      );
      if (existingEdge) return false;

      // TODO: Add more validation rules based on step types

      return true;
    },
    [edges]
  );

  return (
    <div className={styles.pipelineBuilder}>
      <Card className={styles.toolbar}>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => setShowLibrary(true)}
        >
          Add Step
        </Button>
        <Button
          icon={<SaveOutlined />}
          onClick={handleSave}
          style={{ marginLeft: 8 }}
        >
          Save Pipeline
        </Button>
        <Button
          type="primary"
          icon={<PlayCircleOutlined />}
          onClick={handleExecute}
          style={{ marginLeft: 8 }}
        >
          Execute
        </Button>
      </Card>

      <div className={styles.flowContainer}>
        <ReactFlowProvider>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onNodesDelete={onNodesDelete}
            nodeTypes={nodeTypes}
            isValidConnection={isValidConnection}
            fitView
            attributionPosition="bottom-left"
          >
            <Controls />
            <MiniMap
              nodeStrokeColor={(n) => {
                if (n.type === 'connector') return '#52c41a';
                if (n.type === 'transform') return '#1890ff';
                if (n.type === 'llm_process') return '#722ed1';
                if (n.type === 'multimodal') return '#fa8c16';
                if (n.type === 'validate') return '#f5222d';
                return '#d9d9d9';
              }}
              nodeColor={(n) => {
                if (n.type === 'connector') return '#f6ffed';
                if (n.type === 'transform') return '#e6f7ff';
                if (n.type === 'llm_process') return '#f9f0ff';
                if (n.type === 'multimodal') return '#fff7e6';
                if (n.type === 'validate') return '#fff1f0';
                return '#fafafa';
              }}
            />
            <Background variant={'dots' as any} gap={12} size={1} />
          </ReactFlow>
        </ReactFlowProvider>
      </div>

      <Modal
        title="Step Library"
        open={showLibrary}
        onCancel={() => setShowLibrary(false)}
        footer={null}
        width={800}
      >
        <StepLibrary onAddStep={onAddNode} />
      </Modal>

      <Drawer
        title="Step Configuration"
        placement="right"
        open={showConfig}
        onClose={() => setShowConfig(false)}
        width={400}
      >
        {selectedNode && (
          <StepConfigPanel
            node={selectedNode}
            onUpdate={(config) => onUpdateNodeConfig(selectedNode.id, config)}
            onClose={() => setShowConfig(false)}
          />
        )}
      </Drawer>
    </div>
  );
};

export default PipelineBuilder;