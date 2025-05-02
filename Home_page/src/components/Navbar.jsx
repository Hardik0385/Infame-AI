import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Navbar.css';
import MenuIcon from '@mui/icons-material/Menu';
import CloseIcon from '@mui/icons-material/Close';

function Navbar() {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();

  // Handle scroll effect
  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 50) {
        setIsScrolled(true);
      } else {
        setIsScrolled(false);
      }
    };

    window.addEventListener('scroll', handleScroll);
    
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  // Close mobile menu when route changes
  useEffect(() => {
    setIsMenuOpen(false);
  }, [location]);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <nav className={`navbar ${isScrolled ? 'scrolled' : ''} ${isMenuOpen ? 'menu-open' : ''}`}>
      <div className="navbar-content">
        <div className="navbar-logo">
          <Link to="/">
            <img src="/infame_logo.png" alt="InFame Logo" className="logo-image" />
          </Link>
        </div>
        
        <button className="menu-toggle" onClick={toggleMenu}>
          {isMenuOpen ? <CloseIcon /> : <MenuIcon />}
        </button>
        
        <div className={`nav-container ${isMenuOpen ? 'open' : ''}`}>
          <ul className="nav-list">
            <li className="nav-item">
              <Link to="/" className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}>
                <span className="nav-text">Home</span>
              </Link>
            </li>
            <li className="nav-item">
              <Link to="/projects" className={`nav-link ${location.pathname === '/projects' ? 'active' : ''}`}>
                <span className="nav-text">Projects</span>
              </Link>
            </li>
            <li className="nav-item">
              <Link to="/about" className={`nav-link ${location.pathname === '/about' ? 'active' : ''}`}>
                <span className="nav-text">About</span>
              </Link>
            </li>
            <li className="nav-item">
              <Link to="/contact" className={`nav-link ${location.pathname === '/contact' ? 'active' : ''}`}>
                <span className="nav-text">Contact</span>
              </Link>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;