#!/bin/bash

echo "Launching NBA Predictor Frontend..."

if command -v python3 &> /dev/null; then
    echo "Starting local server on http://localhost:8000"
    echo "Press Ctrl+C to stop the server"
    
    python3 -m http.server 8000 &
    
    SERVER_PID=$!
    
    sleep 2
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        #macOS
        open http://localhost:8000
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        #linux
        xdg-open http://localhost:8000
    fi
    
    echo "Server running with PID: $SERVER_PID"
    echo "To stop: kill $SERVER_PID"
    
    wait $SERVER_PID
else
    echo "Python 3 not found. Opening file directly..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open index.html
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open index.html
    fi
fi