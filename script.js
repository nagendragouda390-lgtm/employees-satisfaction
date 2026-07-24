// Smooth scroll for in-page nav links
document.querySelectorAll('a[href^="#"]').forEach(link => {
  link.addEventListener('click', (e) => {
    const target = document.querySelector(link.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

// Show a loading state on the predict button while the form submits
const form = document.getElementById('predict-form');
if (form) {
  form.addEventListener('submit', () => {
    const btn = document.getElementById('submit-btn');
    const label = btn.querySelector('.btn-text');
    btn.disabled = true;
    label.textContent = 'Analyzing profile…';
  });
}

// Animate the risk gauge fill on the result page
window.addEventListener('DOMContentLoaded', () => {
  const fill = document.querySelector('.gauge-fill');
  if (fill) {
    const target = fill.style.width;
    fill.style.width = '0%';
    requestAnimationFrame(() => {
      setTimeout(() => { fill.style.width = target; }, 100);
    });
  }
});
