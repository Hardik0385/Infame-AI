import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom'
import './App.css'
import Navbar from './components/Navbar'
import SplineScene from './components/SplineScene'
import LandingPage from './components/LandingPage'
import Contact from './components/Contact'

// Component to conditionally render the floating phrase based on the route
function MainContent() {
  const location = useLocation();
  const isHomePage = location.pathname === '/';
  
  return (
    <>
      {/* Floating Phrase - only show on home page */}
      {isHomePage && (
        <div className="floating-phrase">
          <p>Viral or Valuable?</p>
        </div>
      )}
      
      {/* Main Content Area */}
      <div className="main-content">
        <Routes>
          <Route path="/" element={
            <>
              <h1>Welcome to InFame!</h1>
              <SplineScene />
            </>
          } />
          <Route path="/contact" element={<Contact />} />
          <Route path="/projects" element={<h1>Projects Page (Coming Soon)</h1>} />
          <Route path="/about" element={<h1>About Page (Coming Soon)</h1>} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </>
  );
}

function App() {
  const [showLandingPage, setShowLandingPage] = useState(true);

  const handleEnterMuseum = () => {
    setShowLandingPage(false);
  };

  if (showLandingPage) {
    return <LandingPage onEnter={handleEnterMuseum} />;
  }

  return (
    <Router>
      {/* Video Background */}
      <video className="video-background" autoPlay loop muted>
        <source src="/video.mp4" type="video/mp4" />
        Your browser does not support the video tag.
      </video>
      
      {/* Dark Overlay */}
      <div className="overlay"></div>
      
      {/* Navigation */}
      <Navbar />
      
      {/* Content with conditional floating phrase */}
      <MainContent />
    </Router>
  )
}

export default App
