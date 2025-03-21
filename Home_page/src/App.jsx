import { useState } from 'react'
import './App.css'
import Navbar from './components/Navbar'
import SplineScene from './components/SplineScene';

function App() {
  const [sidebarVisible, setSidebarVisible] = useState(false);

  // Handle sidebar visibility changes from the Navbar component
  const handleSidebarVisibilityChange = (isVisible) => {
    setSidebarVisible(isVisible);
  };

  return (
    <>
      {/* Video Background */}
      <video className="video-background" autoPlay loop muted>
        <source src="/video.mp4" type="video/mp4" />
        Your browser does not support the video tag.
      </video>
      
      {/* Dark Overlay */}
      <div className="overlay"></div>
      
      <div className="layout">
        {/* Include the Navbar component and pass the visibility handler */}
        <Navbar onSidebarVisibilityChange={handleSidebarVisibilityChange} />

        {/* Main Content Area */}
        <div className={`main-content ${sidebarVisible ? 'sidebar-visible' : ''}`}>
          <h1>Welcome to InFrame</h1>
          <SplineScene />
        </div>
      </div>
    </>
  )
}

export default App
