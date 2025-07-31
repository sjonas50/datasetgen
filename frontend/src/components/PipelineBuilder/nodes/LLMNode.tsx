import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Card, Tag, Tooltip } from 'antd';
import { RobotOutlined, ThunderboltOutlined } from '@ant-design/icons';
import styles from './StepNode.module.css';

interface LLMNodeData {
  label: string;
  config?: {
    type?: string;
    llm_provider?: string;
    llm_model?: string;
    use_extended_thinking?: boolean;
  };
  status?: 'idle' | 'running' | 'completed' | 'failed';
}

const LLMNode: React.FC<NodeProps<LLMNodeData>> = ({ data, selected }) => {
  const processType = data.config?.type || 'enrich';
  const provider = data.config?.llm_provider || 'claude';
  const useExtendedThinking = data.config?.use_extended_thinking;
  const statusClass = data.status ? styles[`status-${data.status}`] : '';

  return (
    <Card
      className={`${styles.stepNode} ${styles.llmNode} ${statusClass} ${selected ? styles.selected : ''}`}
      size="small"
    >
      <Handle
        type="target"
        position={Position.Top}
        className={styles.handle}
      />
      <div className={styles.content}>
        <div className={styles.icon}>
          <RobotOutlined style={{ color: '#722ed1' }} />
          {useExtendedThinking && (
            <Tooltip title="Extended Thinking Enabled">
              <ThunderboltOutlined 
                style={{ 
                  color: '#faad14', 
                  position: 'absolute', 
                  top: -5, 
                  right: -5,
                  fontSize: 12
                }} 
              />
            </Tooltip>
          )}
        </div>
        <div className={styles.label}>{data.label}</div>
        <div className={styles.tags}>
          <Tag color="purple" style={{ fontSize: "12px" }}>
            {provider}
          </Tag>
          <Tag color="geekblue" style={{ fontSize: "12px" }}>
            {processType}
          </Tag>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        className={styles.handle}
      />
    </Card>
  );
};

export default memo(LLMNode);