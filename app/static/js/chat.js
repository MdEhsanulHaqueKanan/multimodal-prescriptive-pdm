// This script will run when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {

    // Get references to the HTML elements we will be interacting with
    const chatHistory = document.getElementById('chat-history');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-chat-btn');

    // --- Function to add a message to the chat history ---
    // This is the UPDATED addMessage function
    const addMessage = (message, sender) => {
        const messageId = `msg-${Date.now()}`; // Create a unique ID for each message
        const messageElement = document.createElement('div');
        messageElement.id = messageId;
        messageElement.classList.add('chat-message', `${sender}-message`);
        
        const p = document.createElement('p');
        p.textContent = message;
        messageElement.appendChild(p);
        
        // --- NEW: Add feedback buttons ONLY to AI messages ---
        if (sender === 'ai' && !message.startsWith("Error:")) {
            const feedbackContainer = document.createElement('div');
            feedbackContainer.classList.add('feedback-container');

            const thumbUpBtn = document.createElement('button');
            thumbUpBtn.classList.add('feedback-btn');
            thumbUpBtn.innerHTML = '👍';
            thumbUpBtn.onclick = () => sendFeedback(messageId, 'positive');

            const thumbDownBtn = document.createElement('button');
            thumbDownBtn.classList.add('feedback-btn');
            thumbDownBtn.innerHTML = '👎';
            thumbDownBtn.onclick = () => sendFeedback(messageId, 'negative');

            feedbackContainer.appendChild(thumbUpBtn);
            feedbackContainer.appendChild(thumbDownBtn);
            messageElement.appendChild(feedbackContainer);
        }
        
        chatHistory.appendChild(messageElement);
        // Scroll to the bottom of the chat history to see the new message
        chatHistory.scrollTop = chatHistory.scrollHeight;
    };

    // --- NEW: Function to send feedback to the backend ---
    const sendFeedback = async (messageId, feedbackType) => {
        console.log(`Sending feedback for ${messageId}: ${feedbackType}`);
        
        try {
            await fetch('/api/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message_id: messageId, feedback: feedbackType }),
            });
            
            // Give the user visual confirmation by disabling the buttons
            const feedbackButtons = document.querySelectorAll(`#${messageId} .feedback-btn`);
            feedbackButtons.forEach(btn => {
                btn.disabled = true;
                btn.style.opacity = '0.5';
                btn.style.cursor = 'default';
            });

        } catch (error) {
            console.error('Error sending feedback:', error);
        }
    };

    // --- Function to handle sending a message to the backend ---
    const sendMessage = async () => {
        const question = chatInput.value.trim();
        if (!question) return; // Don't send empty messages

        // Display the user's message immediately
        addMessage(question, 'user');
        chatInput.value = ''; // Clear the input field
        chatInput.style.height = 'auto'; // Reset textarea height after sending

        // Show a loading indicator
        const loadingElement = document.createElement('div');
        loadingElement.classList.add('chat-message', 'ai-message', 'loading-message');
        loadingElement.innerHTML = '<p><i>Typing...</i></p>';
        chatHistory.appendChild(loadingElement);
        chatHistory.scrollTop = chatHistory.scrollHeight;

        try {
            // Send the question to our Flask backend API
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question }),
            });

            // Remove the loading indicator
            chatHistory.removeChild(loadingElement);

            if (!response.ok) {
                const errorData = await response.json();
                addMessage(`Error: ${errorData.error || 'Failed to get response'}`, 'ai');
                return;
            }

            // Get the AI's answer from the JSON response
            const data = await response.json();
            addMessage(data.answer, 'ai');

        } catch (error) {
            // Handle network errors
            chatHistory.removeChild(loadingElement);
            console.error('Error sending message:', error);
            addMessage('Error: Could not connect to the server.', 'ai');
        }
    };

    // --- Event Listeners ---
    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (event) => {
        // Allow multi-line input with Shift+Enter
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault(); // Prevent default Enter behavior (new line)
            sendMessage();
        }
    });

    chatInput.addEventListener('input', () => {
        // Auto-grow the textarea
        chatInput.style.height = 'auto';
        chatInput.style.height = `${chatInput.scrollHeight}px`;
    });

});