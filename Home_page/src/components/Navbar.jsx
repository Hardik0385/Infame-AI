import { useState, useEffect } from 'react';
import './Navbar.css';

function Navbar({ onSidebarVisibilityChange }) {
  const [sidebarVisible, setSidebarVisible] = useState(false);

  useEffect(() => {
    const handleMouseMove = (e) => {
      // Show sidebar when mouse is within 20px of the left edge
      if (e.clientX <= 20) {
        setSidebarVisible(true);
        if (onSidebarVisibilityChange) onSidebarVisibilityChange(true);
      } else if (e.clientX > 250 && sidebarVisible) {
        // Hide sidebar when mouse moves far from sidebar
        setSidebarVisible(false);
        if (onSidebarVisibilityChange) onSidebarVisibilityChange(false);
      }
    };

    // Add event listener
    document.addEventListener('mousemove', handleMouseMove);

    // Cleanup
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
    };
  }, [sidebarVisible, onSidebarVisibilityChange]);

  return (
    <>
      {/* Invisible trigger area */}
      <div className="sidebar-trigger"></div>
      
      {/* Transparent Sidebar */}
      <div className={`sidebar ${sidebarVisible ? 'active' : ''}`}>
        <div className="sidebar-logo">InFrame</div>
        <nav className="sidebar-nav">
          <ul className="nav-list">
            <li className="nav-item">
              <a href="#" className="nav-link">Home</a>
            </li>
            <li className="nav-item">
              <a href="#" className="nav-link">Contacts</a>
            </li>
            <li className="nav-item">
              <a href="#" className="nav-link">Projects</a>
            </li>
            <li className="nav-item">
              <a href="#" className="nav-link">About</a>
            </li>
          </ul>
        </nav>
      </div>
    </>
  );
}

export default Navbar; 