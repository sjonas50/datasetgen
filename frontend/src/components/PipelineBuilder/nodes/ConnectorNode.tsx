import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Card, Tag } from 'antd';
import { DatabaseOutlined } from '@ant-design/icons';
import styles from './StepNode.module.css';

interface ConnectorNodeData {
  label: string;
  config?: {
    connector_type?: string;
    file_path?: string;
  };
  status?: 'idle' | 'running' | 'completed' | 'failed';
}

const ConnectorNode: React.FC<NodeProps<ConnectorNodeData>> = ({ data, selected }) => {
  const connectorType = data.config?.connector_type || 'unknown';
  const statusClass = data.status ? styles[`status-${data.status}`] : '';

  return (
    <Card
      className={`${styles.stepNode} ${styles.connectorNode} ${statusClass} ${selected ? styles.selected : ''}`}
      size="small"
    >
      <div className={styles.content}>
        <div className={styles.icon}>
          <DatabaseOutlined style={{ color: '#52c41a' }} />
        </div>
        <div className={styles.label}>{data.label}</div>
        <Tag color="green" style={{ fontSize: '12px' }}>
          {connectorType.toUpperCase()}
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

export default memo(ConnectorNode);