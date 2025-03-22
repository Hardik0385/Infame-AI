import React from 'react';
import './LandingPage.css';

function LandingPage({ onEnter }) {
  return (
    <div className="landing-container">
      {/* Video Background */}
      <video className="landing-video-background" autoPlay loop muted>
        <source src="/the-museum.mp4" type="video/mp4" />
        Your browser does not support the video tag.
      </video>
      
      {/* Dark Overlay */}
      <div className="landing-overlay"></div>
      
      <div className="landing-content">
        
        <button className="enter-button" onClick={onEnter}>
          Get started
        </button>
      </div>
    </div>
  );
}

export default LandingPage; 