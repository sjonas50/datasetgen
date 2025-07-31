import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Card, Tag } from 'antd';
import { CheckCircleOutlined, WarningOutlined } from '@ant-design/icons';
import styles from './StepNode.module.css';

interface ValidateNodeData {
  label: string;
  config?: {
    fail_on_error?: boolean;
    rules?: Array<{
      type: string;
      columns?: string[];
    }>;
  };
  status?: 'idle' | 'running' | 'completed' | 'failed';
}

const ValidateNode: React.FC<NodeProps<ValidateNodeData>> = ({ data, selected }) => {
  const failOnError = data.config?.fail_on_error;
  const ruleCount = data.config?.rules?.length || 0;
  const statusClass = data.status ? styles[`status-${data.status}`] : '';

  return (
    <Card
      className={`${styles.stepNode} ${styles.validateNode} ${statusClass} ${selected ? styles.selected : ''}`}
      size="small"
    >
      <Handle
        type="target"
        position={Position.Top}
        className={styles.handle}
      />
      <div className={styles.content}>
        <div className={styles.icon}>
          <CheckCircleOutlined style={{ color: '#52c41a' }} />
          {failOnError && (
            <WarningOutlined 
              style={{ 
                color: '#ff4d4f', 
                position: 'absolute', 
                top: -5, 
                right: -5,
                fontSize: 12
              }} 
            />
          )}
        </div>
        <div className={styles.label}>{data.label}</div>
        <div className={styles.tags}>
          <Tag color="green" style={{ fontSize: "12px" }}>
            {ruleCount} rules
          </Tag>
          {failOnError && (
            <Tag color="red" style={{ fontSize: "12px" }}>
              strict
            </Tag>
          )}
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

export default memo(ValidateNode);