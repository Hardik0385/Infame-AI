import React from 'react';
import { useNavigate } from 'react-router-dom';
import './About.css';

const About = () => {
  const navigate = useNavigate();

  const handleTeamClick = () => {
    navigate('/Team');  
  };

  return (
    <div className="about-container">
      <h1>About InFame AI</h1>
      
      <div className="about-content">
        <div className="about-section">
          <h2>Our Mission</h2>
          <p>
            At InFame AI, we're on a mission to bridge the gap between viral content and valuable 
            innovation. We believe in creating AI solutions that not only capture attention but also 
            provide lasting impact for businesses and individuals alike.
          </p>
        </div>
        
        <div className="about-section">
          <h2>What We Do</h2>
          <p>
            We specialize in developing cutting-edge AI-powered tools and solutions for content 
            creation, analysis, and optimization. Our technology helps creators, marketers, and 
            businesses understand what makes content valuable beyond just virality.
          </p>
        </div>
        
        <div className="about-projects">
          <h2>Our Projects</h2>
          
          <div className="project-card">
            <h3>AI Content Analyzer</h3>
            <p>
              An advanced tool that uses natural language processing to evaluate content quality, 
              potential engagement, and long-term value. It provides actionable insights to improve 
              your content strategy.
            </p>
          </div>
          
          <div className="project-card">
            <h3>Viral Value Predictor</h3>
            <p>
              Our proprietary algorithm that goes beyond predicting virality to estimate the lasting 
              impact and value generation potential of content across different platforms and audiences.
            </p>
          </div>
          
          
        </div>
        
        <div className="about-section">
          <h2>Our Team</h2>
          <p>
            We are a diverse team of AI researchers, data scientists, content strategists, and 
            designers committed to pushing the boundaries of what's possible with artificial 
            intelligence in the content space.
          </p>
        </div>

        <div className="team-button-section">
          <button className="meet-team-button" onClick={handleTeamClick}>
            Meet Our Team
          </button>
        </div>
      </div>
    </div>
  );
};

export default About;