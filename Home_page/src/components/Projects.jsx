import React, { useState } from 'react';
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
          <button className="project-btn">{buttonText}</button>
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
    <div className="influence-chat-container">
      <h3 className="influence-section-title">Chat with InfluenceIQ Assistant</h3>
      
      <div className="influence-chat-messages">
        {messages.map((message, index) => (
          <div 
            key={index} 
            className={message.role === 'user' ? 'influence-user-message' : 'influence-bot-message'}
          >
            <strong>{message.role === 'user' ? 'You' : 'InfluenceIQ Assistant'}:</strong> {message.content}
          </div>
        ))}
        {isLoading && (
          <div className="influence-loading">
            <div className="influence-loading-dots"></div>
          </div>
        )}
      </div>
      
      <form onSubmit={handleSendMessage} className="influence-chat-input">
        <input
          type="text"
          placeholder="Ask about InfluenceIQ..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isLoading}
        />
        <button 
          type="submit" 
          disabled={isLoading || !input.trim()}
          className="influence-send-btn"
        >
          Send
        </button>
      </form>
    </div>
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
    <div className="influence-search-container">
      <h3 className="influence-section-title">Search for an Influencer</h3>
      
      <form onSubmit={handleSearch} className="influence-search-form">
        <input
          type="text"
          placeholder="Enter influencer's name..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          disabled={isLoading}
        />
        <button 
          type="submit" 
          disabled={isLoading || !searchQuery.trim()}
          className="influence-search-btn"
        >
          Search
        </button>
      </form>
      
      {isLoading && (
        <div className="influence-loading-container">
          <div className="influence-loading-spinner"></div>
          <p>Analyzing influencer...</p>
        </div>
      )}
      
      {result && !isLoading && (
        <div className="influence-result">
          <div className="influence-result-header">
            <div className="influence-avatar">{result.name.charAt(0)}</div>
            <h4>{result.name}</h4>
            <div className="influence-score">{result.overall_score}</div>
          </div>
          
          <div className="influence-metrics">
            <div className="influence-metric">
              <div className="influence-metric-value">{result.credibility_score}</div>
              <div className="influence-metric-label">Credibility</div>
            </div>
            <div className="influence-metric">
              <div className="influence-metric-value">{result.longevity_score}</div>
              <div className="influence-metric-label">Longevity</div>
            </div>
            <div className="influence-metric">
              <div className="influence-metric-value">{result.engagement_score}</div>
              <div className="influence-metric-label">Engagement</div>
            </div>
          </div>
          
          <div className="influence-description">
            <h5>About this Influencer</h5>
            <p>{result.description}</p>
          </div>
        </div>
      )}
    </div>
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
    <div className="influence-top-container">
      <h3 className="influence-section-title">Top Influencers</h3>
      
      <div className="influence-cards">
        {influencers.map((influencer, index) => (
          <div key={index} className="influence-card">
            <div className="influence-card-avatar">{influencer.name.charAt(0)}</div>
            <h4>{influencer.name}</h4>
            <div className="influence-card-handle">@{influencer.handle}</div>
            <div className="influence-card-score">{influencer.overall_score}</div>
            <div className="influence-card-metrics">
              <div className="influence-card-metric">
                <div className="influence-card-metric-value">{influencer.component_scores.credibility}</div>
                <div className="influence-card-metric-label">Credibility</div>
              </div>
              <div className="influence-card-metric">
                <div className="influence-card-metric-value">{influencer.component_scores.longevity}</div>
                <div className="influence-card-metric-label">Longevity</div>
              </div>
              <div className="influence-card-metric">
                <div className="influence-card-metric-value">{influencer.component_scores.engagement}</div>
                <div className="influence-card-metric-label">Engagement</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

function Projects() {
  return (
    <div className="projects-container">
      <h1>Our Projects</h1>
      
      <div className="project-item influence-iq-container">
        <h2 className="project-title">InfluenceIQ: AI Rating System</h2>
        <p className="project-description">
          Our advanced rating system uses AI to analyze credibility, longevity, and engagement to provide a comprehensive influence score for individuals in the digital world.
        </p>
        
        <div className="influence-iq-app">
          <div className="influence-iq-section">
            <ChatBox />
          </div>
          
          <div className="influence-iq-section">
            <InfluencerSearch />
          </div>
          
          <div className="influence-iq-section influence-iq-section-full">
            <TopInfluencers />
          </div>
        </div>
      </div>
      
      <div className="project-item">
        <h2 className="project-title">Infame Project Chatbot</h2>
        <p className="project-description">
          Personal assistant chatbot that helps you with your project management tasks. It can provide you with project updates, deadlines, and other important information.
        </p>
        <a href="https://infame-ai.streamlit.app/" target="_blank" rel="noopener noreferrer" className="project-btn-link">
          <button className="project-btn">View Project</button>
        </a>
      </div>
    </div>
  );
}

export default Projects;