import React from 'react';
import './Team.css';
import githubIcon from '../../public/github.svg';
import mailIcon from '../../public/Mail.svg';
import linkedinIcon from '../../public/linkedin.svg';

const Team = () => {
  const teamMembers = [
    {
      name: "Hardik",
      role: "Front-end developer/designer",
      email: "hardiklitop@gmail.com",
      github: "https://github.com/ryugA17"
    },
    {
      name: "Prateek Sinha",
      role: "AI developer",
      email: "ps826105@gmail.com",
      github: "https://github.com/CzPhantom10"
    },
    {
      name: "Hardik Agrawal",
      role: "Front-end developer/designer",
      email: "hardikanilagrawal@gmail.com",
      github: "https://github.com/Hardik0385",
      linkedin: "https://www.linkedin.com/in/ha0385/"
    },
    {
      name: "Utkrisht Parmar",
      role: "Back-end developer",
      email: "utkrisht.parmar@gmail.com",
      github: "https://github.com/Utkrisht2004",
      linkedin: "https://www.linkedin.com/in/utkrisht-parmar-59977928a/"
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
            <div style={{ display: 'flex', gap: '10px', justifyContent: 'center', alignItems: 'center', width: '100%' }}>
              <a href={`mailto:${member.email}?subject=Hello ${member.name}`} title={member.email} style={{ textDecoration: 'none' }}>
                <img src={mailIcon} alt="email" style={{ width: '24px', height: '24px' }} />
              </a>
              <a href={member.github} target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none' }}>
                <img src={githubIcon} alt="github" style={{ width: '24px', height: '24px' }} />
              </a>
              {member.linkedin && (
                <a href={member.linkedin} target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none' }}>
                  <img src={linkedinIcon} alt="linkedin" style={{ width: '24px', height: '24px' }} />
                </a>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Team;
