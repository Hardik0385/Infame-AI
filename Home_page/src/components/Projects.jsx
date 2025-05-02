import React, { useState, useEffect } from 'react';
import './Projects.css';
import { 
  Box, 
  Container, 
  Grid, 
  Typography, 
  Paper,
  Button,
  TextField,
  CircularProgress,
  Avatar,
  Divider,
  Card,
  CardContent
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import SendIcon from '@mui/icons-material/Send';
import PersonIcon from '@mui/icons-material/Person';
import StarIcon from '@mui/icons-material/Star';
import VerifiedIcon from '@mui/icons-material/Verified';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import PeopleIcon from '@mui/icons-material/People';
import LaunchIcon from '@mui/icons-material/Launch';

// Card component
const ProjectCard = ({ title, description, imageUrl, buttonText, buttonLink }) => {
  return (
    <div className="project-card">
      <div className="card-image">
        <img src={imageUrl} alt={title} />
      </div>
      <div className="card-content">
        <h3>{title}</h3>
        <p>{description}</p>
        <a href={buttonLink} target="_blank" rel="noopener noreferrer" className="project-btn-link">
          <button className="project-btn">
            {buttonText}
            <LaunchIcon sx={{ ml: 1, fontSize: '0.9rem' }} />
          </button>
        </a>
      </div>
    </div>
  );
};

// Embedded InfluenceIQ components
const ChatBox = () => {
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState([
    { role: 'assistant', content: "Hi! I'm your InfluenceIQ assistant. How can I help you understand our AI-powered influence ranking system?" }
  ]);

  // Scroll to bottom effect
  useEffect(() => {
    const chatMessages = document.querySelector('.influence-chat-messages');
    if (chatMessages) {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
  }, [messages]);

  const handleSendMessage = (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    
    // Add user message
    setMessages(prev => [...prev, { role: 'user', content: input }]);
    setIsLoading(true);
    
    // Simulate API call
    setTimeout(() => {
      setMessages(prev => [
        ...prev, 
        { 
          role: 'assistant', 
          content: "Thank you for your question about InfluenceIQ! Our AI-powered system analyzes credibility, longevity, and engagement to provide a comprehensive influence score. Would you like to know more about a specific aspect?"
        }
      ]);
      setIsLoading(false);
      setInput('');
    }, 1500);
  };

  return (
    <Paper elevation={4} className="influence-chat-container">
      <Box className="influence-section-title">
        <Typography variant="h5" component="h3">
          Chat with InfluenceIQ Assistant
        </Typography>
      </Box>
      
      <Box className="influence-chat-messages">
        {messages.map((message, index) => (
          <Box
            key={index} 
            className={message.role === 'user' ? 'influence-user-message' : 'influence-bot-message'}
          >
            {message.role === 'assistant' && (
              <Avatar sx={{ 
                width: 24, 
                height: 24, 
                bgcolor: 'primary.main',
                position: 'absolute',
                left: -12,
                top: -12,
                display: { xs: 'none', sm: 'flex' } 
              }}>
                <StarIcon sx={{ fontSize: 14 }} />
              </Avatar>
            )}
            <Typography variant="body1">
              <strong>{message.role === 'user' ? 'You' : 'InfluenceIQ Assistant'}:</strong> {message.content}
            </Typography>
          </Box>
        ))}
        {isLoading && (
          <Box className="influence-loading">
            <CircularProgress size={20} sx={{ color: 'white', mr: 1 }} />
            <Typography variant="body2" sx={{ opacity: 0.7 }}>
              InfluenceIQ is thinking...
            </Typography>
          </Box>
        )}
      </Box>
      
      <form onSubmit={handleSendMessage} className="influence-chat-input">
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Ask about InfluenceIQ..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isLoading}
          size="small"
          sx={{
            '& .MuiOutlinedInput-root': {
              borderRadius: '30px',
              backgroundColor: 'rgba(30, 30, 45, 0.6)',
              '& fieldset': {
                borderColor: 'rgba(255, 255, 255, 0.1)',
              },
              '&:hover fieldset': {
                borderColor: 'rgba(102, 126, 234, 0.3)',
              },
              '&.Mui-focused fieldset': {
                borderColor: 'rgba(102, 126, 234, 0.5)',
              },
            },
            '& .MuiInputBase-input': {
              color: 'white',
            }
          }}
        />
        <Button 
          type="submit" 
          disabled={isLoading || !input.trim()}
          variant="contained"
          className="influence-send-btn"
          endIcon={<SendIcon />}
          sx={{
            backgroundImage: 'linear-gradient(90deg, #667eea, #764ba2)',
            borderRadius: '30px',
            minWidth: '120px',
          }}
        >
          Send
        </Button>
      </form>
    </Paper>
  );
};

const InfluencerSearch = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleSearch = (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    
    setIsLoading(true);
    
    // Simulate API call
    setTimeout(() => {
      setResult({
        name: searchQuery,
        overall_score: 8.7,
        credibility_score: 8.5,
        longevity_score: 8.9,
        engagement_score: 8.6,
        description: `${searchQuery} is a highly influential figure known for their authentic content and consistent engagement with audiences. Their work demonstrates credibility in their field and they've maintained relevance over an extended period.`
      });
      setIsLoading(false);
    }, 2000);
  };

  return (
    <Paper elevation={4} className="influence-search-container">
      <Box className="influence-section-title">
        <Typography variant="h5" component="h3">
          Search for an Influencer
        </Typography>
      </Box>
      
      <form onSubmit={handleSearch} className="influence-search-form">
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Enter influencer's name..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          disabled={isLoading}
          size="small"
          sx={{
            '& .MuiOutlinedInput-root': {
              borderRadius: '30px',
              backgroundColor: 'rgba(30, 30, 45, 0.6)',
              '& fieldset': {
                borderColor: 'rgba(255, 255, 255, 0.1)',
              },
              '&:hover fieldset': {
                borderColor: 'rgba(102, 126, 234, 0.3)',
              },
              '&.Mui-focused fieldset': {
                borderColor: 'rgba(102, 126, 234, 0.5)',
              },
            },
            '& .MuiInputBase-input': {
              color: 'white',
            }
          }}
        />
        <Button 
          type="submit" 
          disabled={isLoading || !searchQuery.trim()}
          variant="contained"
          className="influence-search-btn"
          endIcon={<SearchIcon />}
          sx={{
            backgroundImage: 'linear-gradient(90deg, #667eea, #764ba2)',
            borderRadius: '30px',
            minWidth: '120px',
          }}
        >
          Search
        </Button>
      </form>
      
      {isLoading && (
        <Box className="influence-loading-container">
          <CircularProgress size={40} sx={{ color: '#667eea' }} />
          <Typography variant="body2" sx={{ mt: 2, color: 'rgba(255, 255, 255, 0.7)' }}>
            Analyzing influencer...
          </Typography>
        </Box>
      )}
      
      {result && !isLoading && (
        <Paper elevation={2} className="influence-result">
          <Box className="influence-result-header">
            <Avatar sx={{ 
              width: 50, 
              height: 50, 
              bgcolor: 'transparent',
              backgroundImage: 'linear-gradient(135deg, #667eea, #764ba2)',
            }}>
              {result.name.charAt(0).toUpperCase()}
            </Avatar>
            <Box sx={{ flex: 1 }}>
              <Typography variant="h6">{result.name}</Typography>
            </Box>
            <Box className="influence-score">
              {result.overall_score}
            </Box>
          </Box>
          
          <Box className="influence-metrics">
            <Paper elevation={1} className="influence-metric">
              <Box sx={{ display: 'flex', justifyContent: 'center', mb: 1 }}>
                <VerifiedIcon sx={{ color: '#667eea', fontSize: 18, mr: 0.5 }} />
              </Box>
              <Typography className="influence-metric-value">{result.credibility_score}</Typography>
              <Typography className="influence-metric-label">Credibility</Typography>
            </Paper>
            <Paper elevation={1} className="influence-metric">
              <Box sx={{ display: 'flex', justifyContent: 'center', mb: 1 }}>
                <AccessTimeIcon sx={{ color: '#667eea', fontSize: 18, mr: 0.5 }} />
              </Box>
              <Typography className="influence-metric-value">{result.longevity_score}</Typography>
              <Typography className="influence-metric-label">Longevity</Typography>
            </Paper>
            <Paper elevation={1} className="influence-metric">
              <Box sx={{ display: 'flex', justifyContent: 'center', mb: 1 }}>
                <PeopleIcon sx={{ color: '#667eea', fontSize: 18, mr: 0.5 }} />
              </Box>
              <Typography className="influence-metric-value">{result.engagement_score}</Typography>
              <Typography className="influence-metric-label">Engagement</Typography>
            </Paper>
          </Box>
          
          <Paper elevation={1} className="influence-description">
            <Typography variant="subtitle2" sx={{ mb: 1 }}>About this Influencer</Typography>
            <Divider sx={{ mb: 1.5, bgcolor: 'rgba(255,255,255,0.1)' }} />
            <Typography variant="body2">{result.description}</Typography>
          </Paper>
        </Paper>
      )}
    </Paper>
  );
};

const TopInfluencers = () => {
  const influencers = [
    {
      name: 'Elon Musk',
      handle: 'elonmusk',
      overall_score: 9.5,
      component_scores: {
        credibility: 9.2,
        longevity: 8.8,
        engagement: 9.8
      }
    },
    {
      name: 'Taylor Swift',
      handle: 'taylorswift',
      overall_score: 9.6,
      component_scores: {
        credibility: 9.3,
        longevity: 9.7,
        engagement: 9.8
      }
    },
    {
      name: 'Cristiano Ronaldo',
      handle: 'cristiano',
      overall_score: 9.7,
      component_scores: {
        credibility: 9.5,
        longevity: 9.8,
        engagement: 9.8
      }
    }
  ];

  return (
    <Paper elevation={4} className="influence-top-container">
      <Box className="influence-section-title">
        <Typography variant="h5" component="h3">
          Top Influencers
        </Typography>
      </Box>
      
      <Grid container spacing={2} className="influence-cards">
        {influencers.map((influencer, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <Paper 
              elevation={2} 
              className="influence-card"
              sx={{
                transition: 'transform 0.3s ease, box-shadow 0.3s ease',
                '&:hover': {
                  transform: 'translateY(-5px)',
                  boxShadow: '0 10px 25px rgba(0, 0, 0, 0.2)',
                }
              }}
            >
              <Avatar 
                sx={{ 
                  width: 60, 
                  height: 60, 
                  mb: 1,
                  bgcolor: 'transparent',
                  backgroundImage: 'linear-gradient(135deg, #667eea, #764ba2)',
                }}
              >
                {influencer.name.charAt(0).toUpperCase()}
              </Avatar>
              <Typography variant="h6" sx={{ mb: 0.5 }}>{influencer.name}</Typography>
              <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.6)', mb: 1.5 }}>
                @{influencer.handle}
              </Typography>
              <Box className="influence-card-score">
                {influencer.overall_score}
              </Box>
              <Box className="influence-card-metrics">
                <Box className="influence-card-metric">
                  <Typography className="influence-card-metric-value">
                    {influencer.component_scores.credibility}
                  </Typography>
                  <Typography className="influence-card-metric-label">
                    Credibility
                  </Typography>
                </Box>
                <Box className="influence-card-metric">
                  <Typography className="influence-card-metric-value">
                    {influencer.component_scores.longevity}
                  </Typography>
                  <Typography className="influence-card-metric-label">
                    Longevity
                  </Typography>
                </Box>
                <Box className="influence-card-metric">
                  <Typography className="influence-card-metric-value">
                    {influencer.component_scores.engagement}
                  </Typography>
                  <Typography className="influence-card-metric-label">
                    Engagement
                  </Typography>
                </Box>
              </Box>
            </Paper>
          </Grid>
        ))}
      </Grid>
    </Paper>
  );
};

function Projects() {
  const [fadeIn, setFadeIn] = useState(false);

  useEffect(() => {
    setFadeIn(true);
  }, []);

  return (
    <div className={`projects-container ${fadeIn ? 'fade-in' : ''}`}>
      <Typography 
        variant="h2" 
        component="h1" 
        sx={{ 
          mb: 4,
          textAlign: 'center',
          background: 'linear-gradient(90deg, #667eea, #764ba2)',
          WebkitBackgroundClip: 'text',
          backgroundClip: 'text',
          color: 'transparent',
          fontFamily: '"Press Start 2P", cursive',
          textShadow: '0 2px 10px rgba(102, 126, 234, 0.3)'
        }}
      >
        Our Projects
      </Typography>
      
      <Box className="project-item influence-iq-container">
        <Typography variant="h3" className="project-title">
          InfluenceIQ: AI Rating System
        </Typography>
        <Typography className="project-description">
          Our advanced rating system uses AI to analyze credibility, longevity, and engagement to provide a comprehensive influence score for individuals in the digital world.
        </Typography>
        
        <Box className="influence-iq-app">
          <Box className="influence-iq-section">
            <ChatBox />
          </Box>
          
          <Box className="influence-iq-section">
            <InfluencerSearch />
          </Box>
          
          <Box className="influence-iq-section influence-iq-section-full">
            <TopInfluencers />
          </Box>
        </Box>
      </Box>
      
      <Paper 
        elevation={4} 
        className="project-item"
        sx={{ 
          mt: 4,
          transition: 'transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s',
          '&:hover': {
            transform: 'translateY(-8px)',
            boxShadow: '0 15px 35px rgba(0, 0, 0, 0.3)',
          }
        }}
      >
        <Typography variant="h3" className="project-title">
          Infame Project Chatbot
        </Typography>
        <Typography className="project-description">
          Personal assistant chatbot that helps you with your project management tasks. It can provide you with project updates, deadlines, and other important information.
        </Typography>
        <a href="https://infame-ai.streamlit.app/" target="_blank" rel="noopener noreferrer" className="project-btn-link">
          <Button 
            className="project-btn"
            variant="contained"
            endIcon={<LaunchIcon />}
            sx={{
              backgroundImage: 'linear-gradient(90deg, #667eea, #764ba2)',
              borderRadius: '30px',
              padding: '0.8rem 1.8rem',
              fontFamily: '"Press Start 2P", cursive',
              fontSize: '0.8rem',
              letterSpacing: '1px',
              boxShadow: '0 5px 15px rgba(102, 126, 234, 0.4)',
              transition: 'all 0.3s ease',
              '&:hover': {
                transform: 'translateY(-3px)',
                boxShadow: '0 8px 20px rgba(102, 126, 234, 0.6)',
              },
            }}
          >
            View Project
          </Button>
        </a>
      </Paper>
    </div>
  );
}

export default Projects;