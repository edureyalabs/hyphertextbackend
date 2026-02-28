HTML_KNOWLEDGE = """
SINGLE FILE HTML ARCHITECTURE RULES

A single HTML file is a complete self-contained runtime. Structure every file in this order:
1. DOCTYPE + meta tags + title + Google Fonts import
2. CSS custom properties on :root (all design tokens here)
3. CSS reset and base styles
4. Component styles
5. HTML body with semantic structure
6. JS state object
7. JS render functions
8. JS event handlers
9. JS init call

DESIGN TOKENS PATTERN
Always declare all colors, spacing, fonts as CSS custom properties on :root.
Never hardcode values inside component styles.
Example:
:root {
  --bg: #f8f7f4;
  --surface: #ffffff;
  --text: #111111;
  --muted: #888888;
  --accent: #2563eb;
  --radius: 8px;
  --spacing: 1rem;
  --font-body: 'DM Sans', sans-serif;
  --font-mono: 'DM Mono', monospace;
}

STATE MANAGEMENT PATTERN
One state object, one setState function, one render function.
const state = { ... };
function setState(patch) {
  Object.assign(state, patch);
  render();
}
function render() {
  // pure function: given state, update DOM
}

ROUTING PATTERN FOR SINGLE FILE
Use hash-based routing for zero-config navigation.
window.addEventListener('hashchange', route);
function route() {
  const page = window.location.hash.replace('#', '') || 'home';
  setState({ currentPage: page });
}

CSS LAYOUT PATTERNS
CSS Grid for page-level layout. Flexbox for component-level alignment.
Two-column dashboard: grid-template-columns: 240px 1fr;
Responsive card grid: grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
Centered layout: display:grid; place-items:center; min-height:100vh;

MODERN CSS FEATURES TO USE
:has() for parent-conditional styles — form:has(:invalid) button { opacity: 0.5; }
Container queries for component responsiveness — @container (min-width: 400px) { }
CSS custom properties as reactive state — JS sets --sidebar-open:1, CSS reads it
Scroll-driven animations — animation-timeline: scroll()
View transitions — document.startViewTransition(() => setState(...))
CSS nesting — supported in all modern browsers, use it
clamp() for fluid sizing — font-size: clamp(1rem, 2.5vw, 1.5rem)

COMPONENT COMMUNICATION PATTERN
Use CustomEvent on document for decoupled component messaging.
document.dispatchEvent(new CustomEvent('cart:update', { detail: { count: 3 } }));
document.addEventListener('cart:update', (e) => { ... });

DOM RENDERING PATTERN
Use innerHTML for full component re-renders on small components.
Use targeted element updates for performance-critical parts.
Always batch DOM reads before DOM writes to avoid layout thrash.
Use DocumentFragment for inserting multiple elements.

CDN IMPORTS
Always use esm.sh for ESM imports in script type=module.
Always pin to a specific version, never @latest.
Examples:
import confetti from 'https://esm.sh/canvas-confetti@1.9.2';
import { marked } from 'https://esm.sh/marked@12.0.0';
import Chart from 'https://esm.sh/chart.js@4.4.0/auto';
import * as d3 from 'https://esm.sh/d3@7.9.0';
import Sortable from 'https://esm.sh/sortablejs@1.15.2';
For Three.js use cdnjs: https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js

ACCESSIBILITY RULES
Every interactive element gets a role or semantic tag.
Custom buttons must have role=button and tabindex=0 and keydown Enter handler.
Modals need role=dialog, aria-modal=true, aria-labelledby, focus trap.
Form inputs always have associated label elements.
Color contrast must meet WCAG AA minimum.
Animated elements must respect prefers-reduced-motion.
@media (prefers-reduced-motion: reduce) { * { animation-duration: 0.01ms !important; } }

STORAGE PATTERNS
localStorage for user preferences, theme, simple key-value persistence.
IndexedDB via idb library for structured data, file blobs, larger datasets.
URL params (URLSearchParams) for shareable state.
Never use cookies from JavaScript unless explicitly needed.

FORM PATTERNS
Use the native constraint validation API before writing custom validation.
input.setCustomValidity('message') for custom error states.
form.addEventListener('submit', (e) => { e.preventDefault(); ... });
Always show inline validation feedback, never alert().

ANIMATION PRINCIPLES
Use CSS transitions for state changes. Use CSS keyframes for looping animations.
Use JS (requestAnimationFrame) only for canvas or complex sequence animations.
Entry animations: opacity 0 to 1 + translateY 8px to 0, duration 200-300ms.
Hover transitions: 150ms ease for color/background, 200ms for transform.
Page transitions: use View Transitions API when available.

DARK MODE PATTERN
Add data-theme attribute to html element.
:root[data-theme=dark] { --bg: #0f0f0f; --surface: #1a1a1a; --text: #f0f0f0; }
Toggle: document.documentElement.dataset.theme = isDark ? 'dark' : 'light';
Persist: localStorage.setItem('theme', isDark ? 'dark' : 'light');
Respect system: window.matchMedia('(prefers-color-scheme: dark)').matches

TYPOGRAPHY RULES
Import 2 fonts maximum from Google Fonts. One display/body, one mono.
Use font-weight 300 for body copy, 400-500 for UI labels, 600+ for headings.
Line-height: 1.6 for body text, 1.2 for headings.
Letter-spacing: -0.02em to -0.04em for large headings.
Fluid heading: font-size: clamp(1.5rem, 4vw, 3rem);

ERROR AND LOADING STATES
Every async operation must have three states: loading, success, error.
Show a spinner or skeleton for loading. Never block the entire UI.
Show inline error messages near the failed element, not a global alert.
Use optimistic updates for fast perceived performance.

PERFORMANCE RULES
Images always have loading=lazy unless above the fold.
Use content-visibility: auto on long lists or off-screen sections.
Debounce input event handlers: 300ms for search, 16ms for resize.
Avoid addEventListener inside loops. Use event delegation on parent.
Keep inline scripts below 300 lines total. If bigger, consider splitting logic into clear sections with comments.

WHAT NEVER TO DO
Never use document.write() outside of iframe srcdoc patterns.
Never use inline onclick attributes. Always use addEventListener.
Never manipulate style.property directly. Use CSS classes or custom properties.
Never use var. Always const/let.
Never use synchronous XMLHttpRequest.
Never import React, Vue, Angular, or any component framework. Vanilla only.
"""


def get_knowledge_context() -> str:
    return HTML_KNOWLEDGE.strip()