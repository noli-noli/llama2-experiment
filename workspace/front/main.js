// DOM
const chatForm = document.querySelector('#chat-form');
const chatInput = document.querySelector('#chat-input');
const chatSend = document.querySelector('#chat-send');
const messageContainer = document.querySelector('.messages');
const sendImg = document.querySelector('#send-img');
const loader = document.querySelector('.loader');
const server = 'http://172.17.100.20:8888/items/?text=';
const messages = []; // store previous messages to remember whole conversation

// Function to add a chat message to the container
function addMessage(message, isUser) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');
    messageDiv.textContent = message;
    messageContainer.appendChild(messageDiv);

    // Scroll to the bottom of the chat container
    messageContainer.scrollTop = messageContainer.scrollHeight;
}


// Function to handle user input
function handleUserInput(event) {
    event.preventDefault();
    const message = chatInput.value.trim();
    if (message !== '') {
        messages.push({
            'role': 'user',
            'content': message
        });
        addMessage(message, true);
        fetch(server + message)
        .then(response => response.text())
        // レスポンスが返ってきたら実行する処理
        .then(data => {
            hideLoader();
            addMessage(data, false);
            console.log('success:', data);
        })
        .catch(error => {
            // エラーが発生した場合の処理
            hideLoader();
            addMessage(error, false);
            console.error('error:', error);
        });
        //
        chatInput.value = '';
        showLoader();
    }
}

// Function to show the loader icon
function showLoader() {
    loader.style.display = 'inline-block';
    chatSend.disabled = true;
}

// Function to hide the loader icon
function hideLoader() {
    loader.style.display = 'none';
    chatSend.disabled = false;
}

// Add an event listener to the form
chatForm.addEventListener('submit', handleUserInput);
