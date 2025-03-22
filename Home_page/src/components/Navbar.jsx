import React from 'react';
import './Navbar.css';

function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-logo">
        <img src="/logo_infame.png" alt="InFame Logo" className="logo-image" />
      </div>
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
  );
}

export default Navbar;