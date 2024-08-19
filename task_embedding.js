// #####################################################################################
// ######################### Using Word Embedding / word2vec ###########################
// #####################################################################################
document.addEventListener('DOMContentLoaded', () => {

    async function simplifyInputSentenceEmbedded() {
        console.log("Simplify input sentence function called.");

        // Show processing message
        const processingMessage = "Processing, please wait...";
        document.getElementById('simplified-sentence-embedding').innerText = processingMessage;

        const sentence = document.getElementById('sentence-input-embedding').value;
        document.getElementById('original-sentence-embedding').innerText = sentence;
        console.log("Original sentence:", sentence);

        try {
            const response = await fetch('https://verbapi.zeitconsultancy.com/simplify_sentence', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ sentence }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            const simplifiedSentence = data.simplified_sentence;
            console.log("Simplified sentence:", simplifiedSentence);
            document.getElementById('simplified-sentence-embedding').innerText = simplifiedSentence;
        } catch (error) {
            console.error('There was a problem with the fetch operation:', error);
            document.getElementById('simplified-sentence-embedding').innerText = "An error occurred while processing the sentence.";
        }
    }

    window.simplifyInputSentenceEmbedded = simplifyInputSentenceEmbedded;

});
