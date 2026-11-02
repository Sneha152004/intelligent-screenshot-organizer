import { useState } from 'react'
import Login from './Login'
import Dashboard from './Dashboard'

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(() => {
    return localStorage.getItem('iris_logged_in') !== 'false'
  })

  const handleLogin = () => {
    localStorage.setItem('iris_logged_in', 'true')
    setIsLoggedIn(true)
  }

  const handleLogout = () => {
    localStorage.setItem('iris_logged_in', 'false')
    setIsLoggedIn(false)
  }

  return (
    <>
      {isLoggedIn ? (
        <Dashboard onLogout={handleLogout} />
      ) : (
        <Login onLogin={handleLogin} />
      )}
    </>
  )
}

export default App
