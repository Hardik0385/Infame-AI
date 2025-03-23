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
        <ProjectCard 
          title="AI Content Analyzer" 
          description="Advanced tool that uses natural language processing to evaluate content quality, potential engagement, and long-term value. It provides actionable insights to improve your content strategy."
          imageUrl="/logo_mini.svg"
          buttonText="View Project"
          buttonLink="https://infamefinal.streamlit.app/"
        />
        
        <ProjectCard 
          title="Our Project Personalized Guide" 
          description="This is a project that guides you with whatever ai related queries you have for the first project."
          imageUrl="/bot.svg"
          buttonText="View Project"
          buttonLink="https://infame-ai.streamlit.app/"
          className="bot-image"
        />
      </div>
    </div>
  );
}

export default Projects;