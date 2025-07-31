import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Card } from 'antd';
import {
  DatabaseOutlined,
  FunctionOutlined,
  RobotOutlined,
  FileImageOutlined,
  CheckCircleOutlined,
  QuestionOutlined,
} from '@ant-design/icons';
import styles from './StepNode.module.css';

interface StepNodeData {
  label: string;
  config?: any;
  icon?: string;
  status?: 'idle' | 'running' | 'completed' | 'failed';
}

const iconMap: Record<string, React.ReactNode> = {
  connector: <DatabaseOutlined />,
  transform: <FunctionOutlined />,
  llm_process: <RobotOutlined />,
  multimodal: <FileImageOutlined />,
  validate: <CheckCircleOutlined />,
};

const StepNode: React.FC<NodeProps<StepNodeData>> = ({ data, selected }) => {
  const icon = data.icon ? iconMap[data.icon] : <QuestionOutlined />;
  const statusClass = data.status ? styles[`status-${data.status}`] : '';

  return (
    <Card
      className={`${styles.stepNode} ${statusClass} ${selected ? styles.selected : ''}`}
      size="small"
    >
      <Handle
        type="target"
        position={Position.Top}
        className={styles.handle}
      />
      <div className={styles.content}>
        <div className={styles.icon}>{icon}</div>
        <div className={styles.label}>{data.label}</div>
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        className={styles.handle}
      />
    </Card>
  );
};

export default memo(StepNode);