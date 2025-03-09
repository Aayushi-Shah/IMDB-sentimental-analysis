import React, { useState } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import '../style/SentimentAnalyzer.css';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import TroubleshootIcon from '@mui/icons-material/Troubleshoot';
import {
  Button,
  TextField,
  MenuItem,
  Select,
  InputLabel,
  FormControl,
  Card,
  CardContent,
  Typography,
} from '@mui/material';

// Import Chart.js components and register them
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

interface SentimentResponse {
  sentiment: string;
  model: string;
  probability?: number;
  metrics: {
    accuracy: number;
    precision?: number;
    recall?: number;
    f1?: number;
  };
}

const theme = createTheme({
  typography: {
    h5: {
      color: '#1c2e69',
      fontWeight: 600,
      margin: '1em',
    },
    h6: {
      color: '#1c2e69',
    },
    body1: {
      color: '#1c2e69',
    },
  },
});

const styleButton = {
  color: '#04d488',
  width: '50%',
  border: '#1c2e69 dashed 1px',
  'border-radius': '24px',
  'font-weight': '700',
  '&:hover': {
    border: '#1c2e69 dashed 1px',
  },
};

const models = [
  { value: 'logistic_regression', label: 'Logistic Regression (Classical)' },
  { value: 'naive_bayes', label: 'Naive Bayes (Classical)' },
  { value: 'svm', label: 'SVM (Classical)' },
  { value: 'rnn_lstm', label: 'RNN with LSTM (Deep)' },
  { value: 'bilstm', label: 'BiLSTM (Deep)' },
  { value: 'cnn', label: 'CNN (Deep)' },
];

const SentimentAnalyzer: React.FC = () => {
  const [review, setReview] = useState('');
  const [selectedModel, setSelectedModel] = useState(models[0].value);
  const [responseData, setResponseData] = useState<SentimentResponse | null>(null);
  const [loading, setLoading] = useState(false);

  // Track whether the review field is empty
  const [isReviewEmpty, setIsReviewEmpty] = useState(false);

  const handleSubmit = async () => {
    // Trim the input to avoid whitespace-only strings
    if (!review.trim()) {
      setIsReviewEmpty(true);
      return;
    }
    setIsReviewEmpty(false);

    setLoading(true);
    try {
      const res = await axios.post<SentimentResponse>('http://localhost:8000/predict', {
        review,
        model: selectedModel,
      });
      setResponseData(res.data);
    } catch (error) {
      console.error('Prediction error:', error);
    }
    setLoading(false);
  };

  // Prepare chart data for performance metrics
  const chartData = {
    labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    datasets: [
      {
        label: '',
        data: [
          responseData?.metrics.accuracy || 0,
          responseData?.metrics.precision || 0,
          responseData?.metrics.recall || 0,
          responseData?.metrics.f1 || 0,
        ],
        fill: false,
        borderColor: '#04d488',
      },
    ],
  };

  return (
    <ThemeProvider theme={theme}>
      <Card style={{ maxWidth: 600, margin: '2rem auto', padding: '1rem', backgroundColor: '#f5d1cb' }}>
        <CardContent style={{ backgroundColor: '#fff' }}>
          <Typography variant="h5" align="center">
            Movie Review Sentiment Analyzer
          </Typography>
          <FormControl fullWidth margin="normal">
            <Typography variant="body1">Enter your movie review</Typography>
            <TextField
              multiline
              rows={4}
              value={review}
              onChange={(e) => setReview(e.target.value)}
              error={isReviewEmpty}
              helperText={isReviewEmpty ? 'Review cannot be empty.' : ''}
            />
          </FormControl>
          <FormControl fullWidth margin="normal">
            <Typography variant="body1">Select Model</Typography>
            <Select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
              {models.map((model) => (
                <MenuItem key={model.value} value={model.value}>
                  {model.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <div style={{ display: 'flex', justifyContent: 'center', margin: '1em' }}>
            <Button
              startIcon={<TroubleshootIcon />}
              variant="outlined"
              sx={styleButton}
              onClick={handleSubmit}
              fullWidth
              disabled={loading}
            >
              {loading ? 'Analyzing...' : 'Analyze Sentiment'}
            </Button>
          </div>

          {responseData && (
            <div style={{ marginTop: '2rem' }}>
              <Typography variant="h5" align="center">
                Prediction Result
              </Typography>
              <Typography variant="h6">Sentiment: {responseData.sentiment}</Typography>
              {responseData.probability && (
                <Typography variant="h6">
                  Probability: {responseData.probability.toFixed(2)}
                </Typography>
              )}
              <Typography variant="h6">Model: {responseData.model}</Typography>

              <div style={{ marginTop: '2rem' }}>
                <Typography variant="h5" align="center">
                  Model Performance Metrics
                </Typography>
                <Line style={{ color: '#04d488' }} data={chartData} />
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </ThemeProvider>
  );
};

export default SentimentAnalyzer;
