import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import './App.css'
import Navbar from './components/Navbar'
import SplineScene from './components/SplineScene'
import LandingPage from './components/LandingPage'
import Contact from './components/Contact'
import About from './components/About'
import Team from './components/Team'
import Projects from './components/Projects'

// Page transition animations
const pageVariants = {
  initial: {
    opacity: 0,
    y: 20
  },
  animate: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.6,
      ease: [0.6, 0.05, 0.01, 0.99]
    }
  },
  exit: {
    opacity: 0,
    y: -20,
    transition: {
      duration: 0.4
    }
  }
};

// Animated Routing Component
function AnimatedRoutes() {
  const location = useLocation();
  const isHomePage = location.pathname === '/';
  
  return (
    <>
      {/* Floating Phrase - only show on home page */}
      {isHomePage && (
        <motion.div 
          className="floating-phrase"
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.8, duration: 0.8 }}
        >
          <p>Viral or Valuable?</p>
        </motion.div>
      )}
      
      {/* Main Content Area */}
      <motion.div className="main-content">
        <AnimatePresence mode="wait">
          <Routes location={location} key={location.pathname}>
            <Route path="/" element={
              <motion.div
                initial="initial"
                animate="animate"
                exit="exit"
                variants={pageVariants}
              >
                <h1>Welcome to InFame!</h1>
                <SplineScene />
              </motion.div>
            } />
            <Route path="/contact" element={
              <motion.div
                initial="initial"
                animate="animate"
                exit="exit"
                variants={pageVariants}
              >
                <Contact />
              </motion.div>
            } />
            <Route path="/projects" element={
              <motion.div
                initial="initial"
                animate="animate"
                exit="exit"
                variants={pageVariants}
              >
                <Projects />
              </motion.div>
            } />
            <Route path="/about" element={
              <motion.div
                initial="initial"
                animate="animate"
                exit="exit"
                variants={pageVariants}
              >
                <About />
              </motion.div>
            } />
            <Route path="/Team" element={
              <motion.div
                initial="initial"
                animate="animate"
                exit="exit"
                variants={pageVariants}
              >
                <Team />
              </motion.div>
            } />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </AnimatePresence>
      </motion.div>
    </>
  );
}

function App() {
  const [showLandingPage, setShowLandingPage] = useState(true);
  const [isLoading, setIsLoading] = useState(true);
  const [videoLoaded, setVideoLoaded] = useState(false);
  
  // Loading animation timing
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 2000);
    
    return () => clearTimeout(timer);
  }, []);

  const handleEnterMuseum = () => {
    // Smooth transition out
    document.body.classList.add('page-transition');
    setTimeout(() => {
      setShowLandingPage(false);
      document.body.classList.remove('page-transition');
    }, 600);
  };
  
  const handleVideoLoaded = () => {
    setVideoLoaded(true);
  };

  if (showLandingPage) {
    return <LandingPage onEnter={handleEnterMuseum} />;
  }

  return (
    <>
      {/* Loading overlay */}
      {isLoading && (
        <div className="loading-overlay">
          <div className="loader">
            <div className="loader-circle"></div>
            <div className="loader-text">InFame AI</div>
          </div>
        </div>
      )}
      
      <Router>
        {/* Video Background */}
        <div className={`background-container ${videoLoaded ? 'loaded' : ''}`}>
          <video 
            className="video-background" 
            autoPlay 
            loop 
            muted
            onLoadedData={handleVideoLoaded}
          >
            <source src="/video.mp4" type="video/mp4" />
            Your browser does not support the video tag.
          </video>
          
          {/* Gradient Overlay */}
          <div className="overlay"></div>
        </div>
        
        {/* Navigation */}
        <Navbar />
        
        {/* Content with page transitions */}
        <AnimatedRoutes />
        
        {/* Decorative elements */}
        <div className="corner-decoration top-left"></div>
        <div className="corner-decoration top-right"></div>
        <div className="corner-decoration bottom-left"></div>
        <div className="corner-decoration bottom-right"></div>
      </Router>
    </>
  )
}

export default App
