import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { formatCurrency } from '@/utils/formatters';

interface CostTrendChartProps {
  data: Array<{
    date: string;
    cost: number;
  }>;
}

export default function CostTrendChart({ data }: CostTrendChartProps) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
          dataKey="date" 
          tick={{ fontSize: 12 }}
          tickFormatter={(value) => {
            const date = new Date(value);
            return `${date.getMonth() + 1}/${date.getDate()}`;
          }}
        />
        <YAxis 
          tick={{ fontSize: 12 }}
          tickFormatter={(value) => formatCurrency(value)}
        />
        <Tooltip 
          formatter={(value: number) => formatCurrency(value)}
          labelFormatter={(label) => {
            const date = new Date(label);
            return date.toLocaleDateString();
          }}
        />
        <Legend />
        <Line 
          type="monotone" 
          dataKey="cost" 
          stroke="#8884d8" 
          strokeWidth={2}
          name="Total Cost"
          dot={{ r: 4 }}
          activeDot={{ r: 6 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}