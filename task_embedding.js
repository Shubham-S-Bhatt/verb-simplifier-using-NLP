// #####################################################################################
// ######################### Using Word Embedding / word2vec ###########################
// #####################################################################################

document.addEventListener('DOMContentLoaded', () => {

    async function simplifyInputSentenceEmbedded() {
        console.log("Simplify input sentence function called.");

        // Clear the previous simplified sentence
        document.getElementById('simplified-sentence-embedding').innerText = "";

        
        const sentence = document.getElementById('sentence-input-embedding').value;
        document.getElementById('original-sentence-embedding').innerText = sentence;
        console.log("Original sentence:", sentence);
        
        const response = await fetch('http://127.0.0.1:5000/simplify_sentence', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ sentence }),
        });
    
        const data = await response.json();
        const simplifiedSentence = data.simplified_sentence;
        console.log("Simplified sentence:", simplifiedSentence);
        document.getElementById('simplified-sentence-embedding').innerText = simplifiedSentence;
    }

    window.simplifyInputSentenceEmbedded = simplifyInputSentenceEmbedded;

});