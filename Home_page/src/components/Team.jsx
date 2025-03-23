import React from 'react';
import './Team.css';

const Team = () => {
  const teamMembers = [
    {
      name: "Hardik",
      role: "Front-end developer/designer",
      email: "hardiklitop@gmail.com",
    },
    {
        name: "Prateek Sinha",
        role: "AI developer",
        email: "ps826105@gmail.com",
    },
    {
        name: "Hardik Agrawal",
        role: "Front-end developer/designer",
        email: "hardikanilagrawal@gmail.com",
      },
      {
        name: "Utkrisht Parmar",
        role: "Back-end developer",
        email: "utkrisht.parmar@gmail.com",
      },
  ];

  return (
    <div className="team-container">
      <h1>Our Team</h1>
      <div className="team-grid">
        {teamMembers.map((member, index) => (
          <div key={index} className="team-card" style={{ marginBottom: '30px' }}>
            <h2>{member.name}</h2>
            <h3>{member.role}</h3>
            <p>{member.email}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Team;
