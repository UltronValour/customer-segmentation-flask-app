const form = document.getElementById('segment-form');
const resultCard = document.getElementById('result-card');
const resultBadge = document.getElementById('result-badge');
const resultDesc = document.getElementById('result-desc');
const errorMsg = document.getElementById('error-msg');
const centroidIncome = document.getElementById('centroid-income');
const centroidScore = document.getElementById('centroid-score');
const clusterImage = document.getElementById('cluster-image');
const clusterPlaceholder = document.getElementById('cluster-placeholder');
const visualizationCard = document.getElementById('visualization-card');

function showError(message) {
  errorMsg.textContent = message;
  resultCard.style.display = 'none';
  visualizationCard.style.display = 'none';
}

function showResult(data) {
  errorMsg.textContent = '';

  resultBadge.textContent = data.label;
  resultBadge.className = 'badge ' + data.color.toLowerCase();
  resultDesc.textContent = 'This customer belongs to the ' + data.description + '.';

  centroidIncome.textContent = data.centroid_income.toFixed(2) + ' k$';
  centroidScore.textContent = data.centroid_score.toFixed(2);

  resultCard.style.display = 'block';

  // 🔥 Show graph only after prediction
  visualizationCard.style.display = 'block';
}

form.addEventListener('submit', async function (e) {
  e.preventDefault();

  const income = form.income.value;
  const score = form.score.value;
  const button = form.querySelector('button');

  if (!income || !score) {
    showError('Please fill all fields.');
    return;
  }

  // Hide old results before new prediction
  resultCard.style.display = 'none';
  visualizationCard.style.display = 'none';

  button.textContent = 'Predicting...';
  button.disabled = true;

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ income, score })
    });

    const data = await res.json();

    if (data.success) {
      showResult(data);
    } else {
      showError(data.error || 'Prediction failed.');
    }

  } catch {
    showError('Network error.');
  }

  button.textContent = 'Predict segment';
  button.disabled = false;
});

clusterImage.onerror = function () {
  clusterImage.style.display = 'none';
  clusterPlaceholder.style.display = 'block';
};