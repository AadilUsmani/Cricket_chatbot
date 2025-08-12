# 🏏 Cricket AI Assistant

A production-ready FastAPI application that provides a secure cricket knowledge chatbot using Google's Gemini AI with **specialized expertise domains**.

## ✨ Key Features

- **🔐 Enterprise Security**: API key authentication with timing-attack protection
- **⚡ Smart Rate Limiting**: Redis-backed with in-memory fallback
- **🛡️ Input Sanitisation**: Advanced prompt injection prevention and PII redaction
- **🎯 Cricket Expertise**: 6 specialised knowledge domains with expert-level responses
- **📊 Production Monitoring**: Structured logging, metrics, and health checks
- **🚀 Battle-Tested**: Graceful error handling and automatic recovery

## 🧠 Specialised Cricket Intelligence

### Expert Query Types
- **🏛️ General** - Comprehensive cricket knowledge and culture
- **📋 Rules** - ICC laws, regulations, and official playing conditions
- **🌟 Players** - Career analysis, comparisons, and playing styles
- **📚 History** - Cricket heritage from village greens to modern stadiums
- **📊 Statistics** - Records, analytics, and data-driven insights
- **🎯 Techniques** - Coaching guidance and skill development

Each domain uses specialised AI prompts designed by cricket experts to deliver authoritative, contextual responses.

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/AadilUsmani/Cricket_chatbot
cd cricket-ai-assistant
cp .env.template .env
```

### 2. Configure Environment Variables
Edit `.env` and set:
- `GEMINI_API_KEY`: Your Google Gemini AI API key ([Get one here](https://makersuite.google.com/app/apikey))
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

Visit `http://localhost:8000/docs` for interactive API documentation.

## 📡 API Endpoints

### 🔒 Authentication Required (x-api-key header)
| Endpoint | Method | Description | Query Types Supported |
|----------|--------|-------------|----------------------|
| `/ask` | POST | Ask cricket questions | All 6 specialist domains |
| `/quick-fact` | GET | Get fascinating cricket facts | - |
| `/cricket-joke` | GET | Get cricket humor | - |
| `/cricket-quiz` | GET | Get cricket trivia | - |
| `/prompt-types` | GET | List available query types | - |

### 🌍 Public Endpoints
- `GET /` - API information and status
- `GET /health` - Service health check

## 💡 Usage Examples

### Basic Query
```bash
curl -X POST "https://your-app.onrender.com/ask" \
  -H "x-api-key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain the DRS system in cricket",
    "query_type": "rules"
  }'
```

### Response Format
```json
{
  "answer": "The Decision Review System (DRS) is cricket's technology-assisted umpiring system...",
  "query_type": "rules",
  "request_id": "req_123",
  "processing_time_ms": 1250
}
```

## 🎯 Query Type Guide

| Type | Best For | Example Questions |
|------|----------|-------------------|
| **general** | Broad cricket topics | "Why is cricket popular in India?" |
| **rules** | Laws and regulations | "When is a batsman out LBW?" |
| **players** | Career analysis | "Compare Tendulkar vs Lara" |
| **history** | Cricket heritage | "Tell me about the Bodyline series" |
| **statistics** | Records and data | "Highest Test score ever?" |
| **techniques** | Skill improvement | "How to bowl a perfect yorker?" |

## 🚀 Deployment Options

### Option 1: Render (Recommended)
1. Fork/clone repository to GitHub
2. Connect to [Render](https://render.com)
3. Set environment variables:
   - `GEMINI_API_KEY`
   - `SERVICE_API_KEY`
4. Auto-deploy on git push

### Option 2: Manual Deployment
Compatible with: Railway, Heroku, DigitalOcean App Platform, AWS ECS, Google Cloud Run

**Build Command**: `pip install -r requirements.txt`
**Start Command**: `uvicorn chatbot_cricket_complete:app --host 0.0.0.0 --port $PORT --workers 2`
**Health Check**: `/health`

## 🛡️ Security & Production Features

### Security
- ✅ API key authentication with secure comparison
- ✅ Rate limiting (30 requests/minute, configurable)
- ✅ Prompt injection prevention
- ✅ PII redaction (emails, phones, IDs)
- ✅ CORS protection with allowlist
- ✅ Content length limits
- ✅ Trusted host middleware

### Monitoring & Reliability
- ✅ Structured JSON logging
- ✅ Request ID correlation
- ✅ Performance metrics tracking
- ✅ Graceful error handling
- ✅ Automatic retry logic with exponential backoff
- ✅ Health checks and startup validation

## ⚙️ Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ | - | Google Gemini AI API key |
| `SERVICE_API_KEY` | ✅ | - | API authentication key |
| `ENV` | ❌ | development | Environment (development/production) |
| `REDIS_URL` | ❌ | - | Redis URL for distributed rate limiting |
| `FRONTEND_ORIGINS` | ❌ | localhost:3000 | CORS allowed origins (comma-separated) |
| `RATE_LIMIT_DEFAULT` | ❌ | 30/minute | Rate limit configuration |
| `MAX_CONTENT_LENGTH` | ❌ | 524288 | Max request size (bytes) |
| `LLM_TIMEOUT_SECONDS` | ❌ | 30 | AI model timeout |
| `MAX_RETRY_ATTEMPTS` | ❌ | 3 | Retry attempts for failed requests |

## 🧪 Testing & Development

### Local Testing
```bash
# Run tests
python -m pytest tests/

# Load testing
python scripts/load_test.py

# Manual testing
python scripts/test_prompts.py
```

### API Testing
The application includes comprehensive test coverage for all endpoints and query types.

## 📊 Performance & Scaling

- **Response Time**: ~1-3 seconds per query
- **Throughput**: 30 requests/minute per API key (configurable)
- **Concurrency**: Multi-worker support with async processing
- **Caching**: Redis-backed rate limiting and future response caching
- **Monitoring**: Built-in metrics and health checks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Changelog

### v1.1.0 (Latest)
- ✨ Added specialised prompt templates for all 6 query types
- 🚀 Enhanced cricket expertise with domain-specific knowledge
- 🛡️ Improved security and prompt injection prevention
- 📊 Better structured responses and error handling

### v1.0.0
- 🎉 Initial production release
- 🔐 Complete authentication and security system
- ⚡ Rate limiting and monitoring
- 🏏 Basic cricket knowledge capabilities

## 📞 Support & Links

- **Live API**: [https://cricket-chatbot-fyty.onrender.com](https://cricket-chatbot-fyty.onrender.com)
- **Documentation**: Visit `/docs` endpoint for interactive API docs
- **Issues**: [GitHub Issues](https://github.com/AadilUsmani/Cricket_chatbot/issues)
- **Frontend Demo**: [Vercel App](https://v0-image-analysis-cqcyo7qbf-muhammadaadilusmani-7803s-projects.vercel.app)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for cricket enthusiasts worldwide** 🌍🏏
