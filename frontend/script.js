const API_URL = 'server'; //replace

//loading messages
const loadingMessages = [
    "Dribbling the data... ",
    "Passing to the model... ",
    "Warming up the prediction... ",
    "Crunching the numbers... ",
];

let loadingInterval = null;
let loadingMessageIndex = 0;


async function handleSubmit() {
    const input = document.getElementById('playerNameInput');
    const button = document.getElementById('submitButton');
    const rawInput = input.value;

    button.disabled = true;
    button.textContent = "Loading...";

    showLoading();

    try {
        if (typeof window.submitPlayerNameAndGetPrediction === 'function') {
            const result = await window.submitPlayerNameAndGetPrediction(rawInput);
            handleResponse({
                statusCode: result.statusCode,
                prediction: result.playerResult,
                playerName: result.officialPlayerName,
                opponent: result.teamAgainst,
                gameTime: result.timeAndDateEST,
                errorMessage: result.errorMessage
            });
            return;
        }

        //cleaning 
        const cleanedPlayerName = cleanPlayerString(rawInput);

        if (!cleanedPlayerName) {
            showError("Please enter a valid player name.");
            return;
        }

        const response = await sendToBackend(cleanedPlayerName);

        //response
        handleResponse(response);

    } catch (error) {
        showError("Unable to connect to server. Please try again later.");
        console.error('Connection error:', error);
    } finally {
        //redo button
        button.disabled = false;
        button.textContent = "Submit";
        stopLoadingAnimation();
    }
}

//cleaning function
function cleanPlayerString(input) {
    if (!input) return '';

    let cleaned = input.trim();

    cleaned = cleaned.replace(/\s+/g, ' ');

    cleaned = cleaned.replace(/[\x00-\x1F\x7F-\x9F]/g, '');

    cleaned = cleaned.replace(/[''`]/g, "'");

    if (!/^[a-zA-Z\s\-']+$/.test(cleaned)) {
        return '';
    }

    if (cleaned.length > 64) {
        return cleaned.substring(0, 64);
    }

    return cleaned;
}

//sending to backend (change accordingly)
async function sendToBackend(playerName) {
    const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify({
            playerName: playerName
        })
    });

    const statusCode = response.status;
    const data = await response.json();

    return {
        statusCode: statusCode,
        ...data
    };
}


function showLoading() {
    hideAllSections();
    document.getElementById('loadingSection').classList.remove('hidden');

    loadingMessageIndex = 0;
    updateLoadingMessage();
    loadingInterval = setInterval(updateLoadingMessage, 2000);
}

function updateLoadingMessage() {
    const messageElement = document.getElementById('loadingMessage');
    messageElement.textContent = loadingMessages[loadingMessageIndex];
    loadingMessageIndex = (loadingMessageIndex + 1) % loadingMessages.length;
}

function stopLoadingAnimation() {
    if (loadingInterval) {
        clearInterval(loadingInterval);
        loadingInterval = null;
    }
}

//recieving response and handling (change accordingly)
function handleResponse(response) {
    hideAllSections();

    const { statusCode, prediction, playerName, opponent, gameTime, errorMessage } = response;

    if (statusCode === 200) {
        showResult({ prediction, playerName, opponent, gameTime });
    } else if (statusCode === 404) {
        showError(errorMessage || "Player not found. Please check the spelling and try again.");
    } else if (statusCode === 422) {
        showError(errorMessage || "This player is currently injured. Can't make a prediction right now. Please enter another player.");
    } else {
        showError(errorMessage || "Something went wrong. Please try again.");
    }
}

//results (change accordingly)
function showResult(data) {
    const resultSection = document.getElementById('resultSection');
    const resultContent = document.getElementById('resultContent');

    let html = '';

    if (data.playerName) {
        html += `
            <div class="result-item">
                <span class="result-label">Player</span>
                <span class="result-value">${data.playerName}</span>
            </div>
        `;
    }

    if (data.prediction !== null && data.prediction !== undefined) {
        html += `
            <div class="result-item">
                <span class="result-label">Predicted Points</span>
                <span class="result-value prediction-highlight">${data.prediction.toFixed(1)}</span>
            </div>
        `;
    }

    if (data.opponent) {
        html += `
            <div class="result-item">
                <span class="result-label">Opponent</span>
                <span class="result-value">${data.opponent}</span>
            </div>
        `;
    }

    if (data.gameTime) {
        html += `
            <div class="result-item">
                <span class="result-label">Game Time</span>
                <span class="result-value">${data.gameTime}</span>
            </div>
        `;
    }

    resultContent.innerHTML = html;
    resultSection.classList.remove('hidden');
}

//errors
function showError(message) {
    hideAllSections();

    const errorSection = document.getElementById('errorSection');
    const errorMessageEl = document.getElementById('errorMessage');

    errorMessageEl.textContent = message;
    errorSection.classList.remove('hidden');
}

//new searchs
function resetForm() {
    document.getElementById('playerNameInput').value = '';
    hideAllSections();
    document.getElementById('playerNameInput').focus();
}


function hideAllSections() {
    document.getElementById('loadingSection').classList.add('hidden');
    document.getElementById('resultSection').classList.add('hidden');
    document.getElementById('errorSection').classList.add('hidden');
}

//enter
document.getElementById('playerNameInput').addEventListener('keypress', function (event) {
    if (event.key === 'Enter') {
        handleSubmit();
    }
});

window.addEventListener('load', function () {
    document.getElementById('playerNameInput').focus();
});
