import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Card, Tag } from 'antd';
import { FunctionOutlined } from '@ant-design/icons';
import styles from './StepNode.module.css';

interface TransformNodeData {
  label: string;
  config?: {
    type?: string;
    columns?: string[];
  };
  status?: 'idle' | 'running' | 'completed' | 'failed';
}

const TransformNode: React.FC<NodeProps<TransformNodeData>> = ({ data, selected }) => {
  const transformType = data.config?.type || 'custom';
  const statusClass = data.status ? styles[`status-${data.status}`] : '';

  return (
    <Card
      className={`${styles.stepNode} ${styles.transformNode} ${statusClass} ${selected ? styles.selected : ''}`}
      size="small"
    >
      <Handle
        type="target"
        position={Position.Top}
        className={styles.handle}
      />
      <div className={styles.content}>
        <div className={styles.icon}>
          <FunctionOutlined style={{ color: '#1890ff' }} />
        </div>
        <div className={styles.label}>{data.label}</div>
        <Tag color="blue" style={{ fontSize: "12px" }}>
          {transformType}
        </Tag>
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        className={styles.handle}
      />
    </Card>
  );
};

export default memo(TransformNode);