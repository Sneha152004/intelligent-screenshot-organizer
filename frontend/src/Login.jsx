import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import './Login.css';

const BOOT_LINES = [
  "> INITIALIZING I.R.I.S. CORE...",
  "> LOADING OCR ENGINE v3.2.1...",
  "> SCREENSHOT SCANNER ONLINE",
  "> IMAGE ORGANISATION MODULE READY",
  "> STATUS: AWAITING USER AUTHENTICATION"
];

const generateHexStrings = (count) => {
  return Array.from({ length: count }, () => {
    const val = Math.floor(Math.random() * 65535).toString(16).toUpperCase();
    return val.padStart(4, '0');
  });
};

const Login = ({ onLogin }) => {
  // Boot Sequence State
  const [bootPhase, setBootPhase] = useState(0);
  const [bootComplete, setBootComplete] = useState(false);

  // Form State
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  // Boot sequence effect
  useEffect(() => {
    const t1 = setTimeout(() => setBootPhase(1), 500);
    const t2 = setTimeout(() => setBootPhase(2), 1200);
    const t3 = setTimeout(() => setBootPhase(3), 1900);
    const t4 = setTimeout(() => setBootPhase(4), 2600);
    const t5 = setTimeout(() => setBootPhase(5), 3300);
    const t6 = setTimeout(() => setBootComplete(true), 4200);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
      clearTimeout(t4);
      clearTimeout(t5);
      clearTimeout(t6);
    };
  }, []);

  // Matrix Rain
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    const fontSize = 20;
    const columns = Math.floor(canvas.width / fontSize);
    const drops = Array.from({ length: columns }, () => Math.random() * -100);
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';

    let animId;
    const draw = () => {
      ctx.fillStyle = 'rgba(5, 8, 22, 0.05)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.font = `${fontSize}px monospace`;

      for (let i = 0; i < drops.length; i++) {
        const char = chars[Math.floor(Math.random() * chars.length)];
        const x = i * fontSize;
        const y = drops[i] * fontSize;

        if (Math.random() > 0.3) {
          ctx.fillStyle = 'rgba(0, 212, 255, 0.18)';
        } else {
          ctx.fillStyle = 'rgba(0, 102, 255, 0.28)';
        }
        ctx.fillText(char, x, y);

        if (y > canvas.height && Math.random() > 0.975) {
          drops[i] = 0;
        }
        drops[i] += 0.5;
      }
      animId = requestAnimationFrame(draw);
    };
    draw();

    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animId);
    };
  }, []);

  // Particle System
  const particles = useMemo(() => {
    return Array.from({ length: 20 }, (_, i) => ({
      id: i,
      left: Math.random() * 100,
      duration: 10 + Math.random() * 15,
      delay: Math.random() * 15,
      size: 1 + Math.random() * 2
    }));
  }, []);

  // Data streams
  const leftStream = useMemo(() => generateHexStrings(15), []);
  const rightStream = useMemo(() => generateHexStrings(15), []);

  const handleSubmit = (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    setTimeout(() => {
      const cleanUser = username.trim().toLowerCase();
      const cleanPhone = username.trim().replace(/\D/g, '');
      const cleanPass = password.trim();

      const isAuthorizedUser = (cleanUser === 'group@gmail.com' || cleanPhone === '1234567890' || cleanUser === '1234567890');
      const isAuthorizedPass = (cleanPass === 'majorproject');

      if (isAuthorizedUser && isAuthorizedPass) {
        setIsLoading(false);
        if (onLogin) onLogin();
      } else {
        setIsLoading(false);
        if (!cleanUser || !cleanPass) {
          setError('EMAIL/PHONE AND PASSWORD ARE REQUIRED');
        } else {
          setError('ACCESS DENIED: INVALID AUTHORIZED CREDENTIALS');
        }
      }
    }, 900);
  };

  const handleInputChange = (setter) => (e) => {
    setter(e.target.value);
    if (error) setError('');
  };

  return (
    <div className="jarvis-wrapper" id="jarvis-wrapper-id">
      <div className={`jarvis-boot-overlay ${bootComplete ? 'boot-complete' : ''}`} id="jarvis-boot-overlay-id">
        <div className="boot-content">
          <div className="boot-lines">
            {BOOT_LINES.map((line, index) => {
              if (index >= bootPhase) return null;
              const isLast = index === bootPhase - 1 || (bootPhase >= 5 && index === 4);
              return (
                <div 
                  key={index} 
                  className={`boot-line ${isLast && bootPhase < 6 ? 'typing-cursor' : ''}`}
                  style={{ animationDelay: `${index * 0.2}s` }}
                >
                  {line}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <canvas ref={canvasRef} className="matrix-rain" id="matrix-rain-id"></canvas>

      <div className="scanline" id="scanline-id"></div>

      <div className="particles-container" id="particles-container-id">
        {particles.map((p) => (
          <div
            key={p.id}
            className="particle"
            style={{
              left: `${p.left}%`,
              animationDuration: `${p.duration}s`,
              animationDelay: `${p.delay}s`,
              width: `${p.size}px`,
              height: `${p.size}px`
            }}
          ></div>
        ))}
      </div>

      <div className="data-stream left" id="data-stream-left-id">
        {leftStream.map((hex, i) => (
          <span key={`l-${i}`}>{hex}</span>
        ))}
      </div>
      
      <div className="data-stream right" id="data-stream-right-id">
        {rightStream.map((hex, i) => (
          <span key={`r-${i}`}>{hex}</span>
        ))}
      </div>

      {bootComplete && (
        <div className="hud-ring-container" id="hud-ring-container-id">
          <div className="hud-ring hud-ring-outer"></div>
          <div className="hud-ring hud-ring-middle"></div>
          <div className="hud-ring hud-ring-inner"></div>
          <div className="hud-core" id="hud-core-id"></div>
        </div>
      )}

      <div className={`jarvis-login-container ${bootComplete ? 'visible' : ''}`} id="jarvis-login-container-id">
        <div className="jarvis-icon" id="jarvis-icon-id">
          <svg viewBox="0 0 100 100" width="80" height="80" style={{ position: 'relative', zIndex: 2 }}>
            <circle cx="50" cy="50" r="35" fill="none" stroke="#00d4ff" strokeWidth="0.5" opacity="0.3" />
            <circle cx="50" cy="50" r="28" fill="none" stroke="#00d4ff" strokeWidth="0.8" strokeDasharray="8 4" className="icon-ring-1" />
            <circle cx="50" cy="50" r="20" fill="none" stroke="#00d4ff" strokeWidth="0.5" strokeDasharray="4 8" className="icon-ring-2" />
            <circle cx="50" cy="50" r="12" fill="none" stroke="#00d4ff" strokeWidth="1" />
            <circle cx="50" cy="50" r="5" fill="#00d4ff" opacity="0.8">
              <animate attributeName="opacity" values="0.4;1;0.4" dur="2s" repeatCount="indefinite" />
            </circle>
          </svg>
        </div>

        <h1 className="jarvis-title">I.R.I.S.</h1>
        <p className="jarvis-subtitle">Intelligent Recognition & Image Sorting</p>

        <form className="jarvis-form" onSubmit={handleSubmit} id="jarvis-login-form-id">
          <div className="jarvis-input-group">
            <label htmlFor="jarvis-username">EMAIL OR PHONE</label>
            <input 
              type="text" 
              id="jarvis-username" 
              placeholder="Enter your email or phone" 
              value={username} 
              onChange={handleInputChange(setUsername)} 
              autoComplete="off" 
            />
          </div>
          
          <div className="jarvis-input-group">
            <label htmlFor="jarvis-password">PASSWORD</label>
            <div className="password-wrapper">
              <input 
                type={showPassword ? 'text' : 'password'} 
                id="jarvis-password" 
                placeholder="Enter your password" 
                value={password} 
                onChange={handleInputChange(setPassword)} 
              />
              <button 
                type="button" 
                className={`password-toggle ${showPassword ? 'active' : ''}`}
                id="password-toggle-btn"
                onClick={() => setShowPassword(!showPassword)}
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.5">
                  {showPassword ? (
                    <>
                      <circle cx="12" cy="12" r="8" strokeDasharray="3 2" />
                      <circle cx="12" cy="12" r="3" fill="currentColor" stroke="none" />
                      <line x1="4" y1="4" x2="20" y2="20" strokeWidth="2" />
                    </>
                  ) : (
                    <>
                      <circle cx="12" cy="12" r="8" strokeDasharray="3 2" />
                      <circle cx="12" cy="12" r="4" />
                      <circle cx="12" cy="12" r="1.5" fill="currentColor" stroke="none" />
                      <line x1="12" y1="2" x2="12" y2="4" />
                      <line x1="12" y1="20" x2="12" y2="22" />
                      <line x1="2" y1="12" x2="4" y2="12" />
                      <line x1="20" y1="12" x2="22" y2="12" />
                    </>
                  )}
                </svg>
              </button>
            </div>
          </div>
          
          {error && (
            <div className="error-message" id="jarvis-error-message-id">
              <span>⚠️</span> {error}
            </div>
          )}
          
          <button type="submit" className={`jarvis-btn ${isLoading ? 'loading' : ''}`} id="jarvis-submit-btn">
            {isLoading ? <span className="btn-spinner"></span> : "SIGN IN"}
          </button>
        </form>

        <div className="status-indicator" id="jarvis-status-indicator-id">
          <span className="status-dot"></span>
          <span className="status-text">OCR ENGINE ONLINE</span>
        </div>
      </div>
    </div>
  );
};

export default Login;
