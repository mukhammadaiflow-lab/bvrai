# =============================================================================
# BVRAI Frontend - Production Dockerfile
# =============================================================================

FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies with legacy peer deps to handle React 19 compatibility
RUN npm install --legacy-peer-deps

# Copy source code
COPY . .

# Build the application
ARG NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

RUN npm run build

# Expose port
EXPOSE 3000

# Start the application
CMD ["npm", "start"]
