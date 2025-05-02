import React, { useState, useEffect } from 'react';
import './LandingPage.css';

function LandingPage({ onEnter }) {
  const [loaded, setLoaded] = useState(false);
  const [showTitle, setShowTitle] = useState(false);
  const [showSubtitle, setShowSubtitle] = useState(false);
  const [showButton, setShowButton] = useState(false);

  useEffect(() => {
    setLoaded(true);
    
    const titleTimer = setTimeout(() => setShowTitle(true), 800);
    const subtitleTimer = setTimeout(() => setShowSubtitle(true), 1600);
    const buttonTimer = setTimeout(() => setShowButton(true), 2400);
    
    return () => {
      clearTimeout(titleTimer);
      clearTimeout(subtitleTimer);
      clearTimeout(buttonTimer);
    };
  }, []);

  return (
    <div className={`landing-container ${loaded ? 'loaded' : ''}`}>
      {/* Video Background */}
      <video className="landing-video-background" autoPlay loop muted>
        <source src="/the-museum.mp4" type="video/mp4" />
        Your browser does not support the video tag.
      </video>
      
      {/* Dark Overlay with Gradient */}
      <div className="landing-overlay"></div>
      
      {/* Animated Particles */}
      <div className="particles">
        {[...Array(20)].map((_, i) => (
          <div key={i} className="particle"></div>
        ))}
      </div>
      
      <div className="landing-content">
        <div className={`landing-title-container ${showTitle ? 'visible' : ''}`}>
          <h1 className="landing-title">InFame AI</h1>
          <div className="title-underline"></div>
        </div>
        
        <div className={`landing-subtitle-container ${showSubtitle ? 'visible' : ''}`}>
          <p className="landing-subtitle">Discover the true influence in the digital world</p>
        </div>
        
        <div className={`landing-button-container ${showButton ? 'visible' : ''}`}>
          <button className="enter-button" onClick={onEnter}>
            <span className="button-text">Get Started</span>
            <span className="button-icon">→</span>
          </button>
        </div>
      </div>
      
      {/* Decorative Elements */}
      <div className="corner-decoration top-left"></div>
      <div className="corner-decoration top-right"></div>
      <div className="corner-decoration bottom-left"></div>
      <div className="corner-decoration bottom-right"></div>
    </div>
  );
}

export default LandingPage; 