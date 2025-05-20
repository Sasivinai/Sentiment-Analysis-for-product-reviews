// Main JavaScript file for ChatGPT Sentiment Analysis application

document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const analyzeBtn = document.getElementById('analyze-btn');
    const reviewTextarea = document.getElementById('review-text');
    const resultContainer = document.getElementById('result-container');
    const originalTextElement = document.getElementById('original-text');
    const cleanedTextElement = document.getElementById('cleaned-text');
    const sentimentBadge = document.getElementById('sentiment-badge');
    
    // Add event listener to the analyze button
    analyzeBtn.addEventListener('click', analyzeSentiment);
    
    // Function to analyze sentiment
    function analyzeSentiment() {
        const text = reviewTextarea.value.trim();
        
        // Check if text is empty
        if (!text) {
            alert('Please enter some text to analyze.');
            return;
        }
        
        // Display loading state
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';
        
        // Send request to the backend
        fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Display results
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error analyzing sentiment. Please try again.');
        })
        .finally(() => {
            // Reset button state
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = 'Analyze Sentiment';
        });
    }
    
    // Function to display results
    function displayResults(data) {
        // Set text content
        originalTextElement.textContent = data.text;
        cleanedTextElement.textContent = data.cleaned_text;
        
        // Set sentiment badge color and text
        sentimentBadge.textContent = data.sentiment.toUpperCase();
        
        // Remove all classes first
        sentimentBadge.classList.remove('positive', 'neutral', 'negative');
        
        // Add appropriate class based on sentiment
        sentimentBadge.classList.add(data.sentiment.toLowerCase());
        
        // Show result container with animation
        resultContainer.classList.remove('d-none');
        resultContainer.classList.add('fade-in');
        
        // Scroll to results
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
});

// For dashboard charts responsiveness
window.addEventListener('resize', function() {
    // Check if we're on the dashboard page
    if (document.getElementById('sentiment-dist-chart')) {
        const charts = [
            'sentiment-dist-chart',
            'rating-dist-chart',
            'platform-dist-chart',
            'language-dist-chart',
            'rating-by-title-chart'
        ];
        
        // Resize all charts
        charts.forEach(chartId => {
            const chart = document.getElementById(chartId);
            if (chart) {
                Plotly.relayout(chart, {
                    autosize: true
                });
            }
        });
    }
});
