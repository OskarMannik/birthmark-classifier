document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const imageInput = document.getElementById('image');
    const imagePreview = document.getElementById('image-preview');
    const results = document.getElementById('results');
    const aboutBtn = document.getElementById('about-btn');
    const modal = document.getElementById('about-modal');
    const closeBtn = document.querySelector('.close-btn');

    modal.style.display = 'none';

    aboutBtn.addEventListener('click', function() {
        modal.style.display = 'flex';
    });

    closeBtn.addEventListener('click', function() {
        modal.style.display = 'none';
    });

    window.addEventListener('click', function(event) {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });

    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
            };
            reader.readAsDataURL(file);
        }
    });

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(form);
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                document.getElementById('prediction').textContent = 
                    `Predicted Class: ${data.prediction}`;
                
                const probsHtml = Object.entries(data.probabilities)
                    .map(([label, prob]) => `<p>${label}: ${(prob * 100).toFixed(2)}%</p>`)
                    .join('');
                
                document.getElementById('probabilities').innerHTML = probsHtml;
                results.classList.remove('hidden');
            } else {
                alert(data.error || 'An error occurred');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while processing your request');
        }
    });
}); 

