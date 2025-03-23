import React, { useState } from 'react';
import Spline from '@splinetool/react-spline';
import './SplineScene.css';

export default function SplineScene() {
  const [isLoading, setIsLoading] = useState(true);

  const handleLoad = () => {
    setIsLoading(false);
  };

  return (
    <div className="spline-container">
      {isLoading && <div className="loading">Loading 3D Scene...</div>}
      <Spline 
        scene="https://prod.spline.design/NLDIwWsnnDTCtMVV/scene.splinecode" 
        onLoad={handleLoad}
      />
    </div>
  );
}
