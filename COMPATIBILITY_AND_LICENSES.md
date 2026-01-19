# Compatibility & Licenses

This document describes the licensing of this project and its dependencies.

## Project License

This project is licensed under the **MIT License**. See [LICENSE](./LICENSE) for details.

## Third-Party Dependencies

### Node.js Services (Token Service, Media Bridge)

All dependencies are permissively licensed (MIT, Apache 2.0, ISC, BSD):

| Package | License | Purpose |
|---------|---------|---------|
| express | MIT | Web framework |
| jsonwebtoken | MIT | JWT signing/verification |
| ws | MIT | WebSocket implementation |
| zod | MIT | Schema validation |
| pino | MIT | Structured logging |
| helmet | MIT | Security headers |
| cors | MIT | CORS middleware |
| express-rate-limit | MIT | Rate limiting |
| uuid | MIT | UUID generation |
| axios | MIT | HTTP client |

### Python Services (Dialog Manager, Ingestion Service)

All dependencies are permissively licensed (MIT, Apache 2.0, BSD):

| Package | License | Purpose |
|---------|---------|---------|
| fastapi | MIT | Web framework |
| uvicorn | BSD-3-Clause | ASGI server |
| pydantic | MIT | Data validation |
| httpx | BSD-3-Clause | HTTP client |
| structlog | Apache 2.0 | Structured logging |
| beautifulsoup4 | MIT | HTML parsing |
| numpy | BSD-3-Clause | Numerical operations |
| aiosqlite | MIT | Async SQLite |

## No GPL Code

This project does **NOT** include any GPL-licensed code. All implementations are original.

## Verification

To verify dependency licenses:

### Node.js
```bash
npx license-checker --summary
```

### Python
```bash
pip-licenses --format=markdown
```

## Optional Dependencies

When integrating production services, verify their licenses:

- **Pinecone**: Proprietary (cloud service)
- **OpenAI**: Proprietary API (MIT client library)
- **Anthropic**: Proprietary API (MIT client library)

## Contributing

When contributing code:
1. Do not copy GPL-licensed code
2. Ensure new dependencies are MIT/Apache 2.0/BSD compatible
3. Document any new dependencies in this file
