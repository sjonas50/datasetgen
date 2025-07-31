import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface ExecutionTrendChartProps {
  data: Array<{
    date: string;
    total: number;
    successful: number;
    failed: number;
  }>;
}

export default function ExecutionTrendChart({ data }: ExecutionTrendChartProps) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
          dataKey="date" 
          tick={{ fontSize: 12 }}
          tickFormatter={(value) => {
            const date = new Date(value);
            return `${date.getMonth() + 1}/${date.getDate()}`;
          }}
        />
        <YAxis tick={{ fontSize: 12 }} />
        <Tooltip 
          labelFormatter={(label) => {
            const date = new Date(label);
            return date.toLocaleDateString();
          }}
        />
        <Legend />
        <Bar 
          dataKey="successful" 
          stackId="a" 
          fill="#52c41a" 
          name="Successful"
        />
        <Bar 
          dataKey="failed" 
          stackId="a" 
          fill="#ff4d4f" 
          name="Failed"
        />
      </BarChart>
    </ResponsiveContainer>
  );
}