import { useState } from 'react'
import './App.css'
import Navbar from './components/Navbar'
import SplineScene from './components/SplineScene'
import LandingPage from './components/LandingPage'

function App() {
  const [showLandingPage, setShowLandingPage] = useState(true);

  const handleEnterMuseum = () => {
    setShowLandingPage(false);
  };

  return (
    <>
      {showLandingPage ? (
        <LandingPage onEnter={handleEnterMuseum} />
      ) : (
        <>
          {/* Video Background */}
          <video className="video-background" autoPlay loop muted>
            <source src="/video.mp4" type="video/mp4" />
            Your browser does not support the video tag.
          </video>
          
          {/* Dark Overlay */}
          <div className="overlay"></div>
          
          {/* Navigation */}
          <Navbar />
          
          {/* Floating Phrase */}
          <div className="floating-phrase">
            <p>Viral or Valuable?</p>
          </div>
          
          {/* Main Content Area */}
          <div className="main-content">
            <h1>Welcome to InFame!</h1>
            <SplineScene />
          </div>
        </>
      )}
    </>
  )
}

export default App
