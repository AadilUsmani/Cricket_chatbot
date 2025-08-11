# 🏏 Cricket AI Assistant

A production-ready FastAPI application that provides a secure cricket knowledge chatbot using Google's Gemini AI.

## Features

- **🔐 Secure Authentication**: API key-based authentication with secure comparison
- **⚡ Rate Limiting**: Redis-backed or in-memory fallback rate limiting
- **🛡️ Input Sanitization**: Prevents prompt injection and basic PII redaction
- **🎯 Cricket Expertise**: Specialized cricket knowledge with multiple query types
- **📊 Comprehensive Logging**: Structured JSON logging for monitoring
- **🚀 Production Ready**: Health checks, error handling, and graceful startup/shutdown

## Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd cricket-ai-assistant
cp .env.template .env
```

### 2. Configure Environment Variables
Edit `.env` and set:
- `GEMINI_API_KEY`: Your Google Gemini AI API key
- `SERVICE_API_KEY`: Generate a strong random key for API authentication
- `FRONTEND_ORIGINS`: Your frontend domain(s)

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Locally
```bash
uvicorn chatbot_cricket_complete:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Authentication Required (x-api-key header)
- `POST /ask` - Ask cricket questions
- `GET /quick-fact` - Get cricket facts
- `GET /cricket-joke` - Get cricket jokes  
- `GET /cricket-quiz` - Get cricket quiz
- `GET /prompt-types` - Available query types

### Public Endpoints
- `GET /` - API information
- `GET /health` - Health check

## Query Types
- `general` - General cricket questions
- `rules` - Cricket rules and regulations
- `players` - Player information
- `history` - Cricket history
- `statistics` - Records and stats
- `techniques` - Playing techniques

## Deployment to Render

### Option 1: Using render.yaml (Recommended)
1. Push code to GitHub
2. Connect repository to Render
3. Set environment variables in Render dashboard:
   - `GEMINI_API_KEY`
   - `SERVICE_API_KEY` 
4. Deploy automatically

### Option 2: Manual Setup
1. Create new Web Service in Render
2. Connect your GitHub repository
3. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn chatbot_cricket_complete:app --host 0.0.0.0 --port $PORT --workers 2`
   - **Health Check Path**: `/health`
4. Set environment variables
5. Deploy

## Security Features

- API key authentication with timing-attack protection
- Rate limiting (30 requests/minute default)
- Input sanitization against prompt injection
- Basic PII redaction (emails, phones, long digit sequences)  
- CORS configuration with allowed origins
- Content length limits
- Comprehensive request logging

## Monitoring

The application provides:
- Health check endpoint at `/health`
- Structured JSON logging
- Request ID tracking
- Performance metrics
- Error tracking with request correlation

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes | - | Google Gemini AI API key |
| `SERVICE_API_KEY` | Yes | - | API authentication key |
| `ENV` | No | development | Environment (development/production) |
| `REDIS_URL` | No | - | Redis URL for rate limiting |
| `FRONTEND_ORIGINS` | No | http://localhost:3000 | CORS allowed origins |
| `RATE_LIMIT_DEFAULT` | No | 30/minute | Rate limit configuration |
| `MAX_CONTENT_LENGTH` | No | 524288 | Max request size in bytes |
| `LLM_TIMEOUT_SECONDS` | No | 30 | LLM request timeout |

## License

This project is licensed under the MIT License.