#!/bin/bash

# Kill any existing processes on ports 8000 and 3000
echo "🧹 Cleaning up ports..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:3000 | xargs kill -9 2>/dev/null

echo "🚀 Starting Backend (Port 8000)..."
nohup ./.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!

# Start Frontend
echo "🌐 Starting Frontend (Port 3000)..."
cd frontend
nohup python3 -m http.server 3000 > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo "✅ Application started!"
echo "   - Frontend: http://localhost:3000"
echo "   - Backend:  http://localhost:8000"
echo "   - PIDs: Backend=$BACKEND_PID, Frontend=$FRONTEND_PID"
echo "   - Logs: backend.log, frontend.log"
