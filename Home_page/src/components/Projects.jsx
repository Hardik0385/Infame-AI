import React from 'react';
import './Projects.css';

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

function Projects() {
  return (
    <div className="projects-container">
      <h1>Our Projects</h1>
      
      <div className="projects-content">
        <div className="project-item">
          <h2 className="project-title">AI Rating System</h2>
          <p className="project-description">
            An advanced tool that uses natural language processing to evaluate content quality, potential engagement, and long-term value. It provides actionable insights to improve your content strategy.
          </p>
          <a href="https://infamefinal.streamlit.app/" target="_blank" rel="noopener noreferrer" className="project-btn-link">
            <button className="project-btn">View Project</button>
          </a>
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
    </div>
  );
}

export default Projects;