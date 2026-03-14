# boilerplate.py

INITIAL_BOILERPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Untitled</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #f8f7f4;
      font-family: 'DM Sans', 'Helvetica Neue', sans-serif;
    }
    .placeholder {
      text-align: center;
      color: #bbb;
      max-width: 320px;
      padding: 2rem;
    }
    .placeholder-icon {
      width: 48px;
      height: 48px;
      margin: 0 auto 1.25rem;
      opacity: 0.25;
    }
    .placeholder h2 {
      font-size: 0.95rem;
      font-weight: 400;
      color: #aaa;
      margin-bottom: 0.85rem;
      letter-spacing: -0.01em;
    }
    .placeholder-hints {
      display: flex;
      flex-direction: column;
      gap: 0.45rem;
      list-style: none;
    }
    .placeholder-hints li {
      font-size: 0.72rem;
      font-family: 'DM Mono', monospace;
      color: #ccc;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.4rem;
    }
    .placeholder-hints li::before {
      content: '';
      width: 3px;
      height: 3px;
      border-radius: 50%;
      background: #ddd;
      flex-shrink: 0;
    }
  </style>
</head>
<body>
  <div class="placeholder">
    <svg class="placeholder-icon" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="4" y="8" width="40" height="32" rx="4" stroke="#999" stroke-width="2"/>
      <path d="M4 16h40" stroke="#999" stroke-width="2"/>
      <circle cx="10" cy="12" r="1.5" fill="#999"/>
      <circle cx="15" cy="12" r="1.5" fill="#999"/>
      <circle cx="20" cy="12" r="1.5" fill="#999"/>
      <path d="M14 26h20M14 31h14" stroke="#ccc" stroke-width="2" stroke-linecap="round"/>
    </svg>
    <h2>describe your page in the chat</h2>
    <ul class="placeholder-hints">
      <li>attach images, logos or documents</li>
      <li>ask for a landing page, portfolio, app</li>
      <li>request edits at any time.</li>
    </ul>
  </div>
</body>
</html>"""