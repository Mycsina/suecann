# Stage 1: Build frontend (WASM is pre-built and committed)
# NOTE: the frontend lives at code/frontend/ since the repo restructure (c426825);
# the build context is the repo root (see deploy/docker-compose.yml).
FROM oven/bun:alpine AS builder

WORKDIR /build
COPY code/frontend/ frontend/
RUN cd frontend && bun install && bun x vite build

# Stage 2: Serve static files
FROM nginx:alpine

COPY --from=builder /build/frontend/dist /usr/share/nginx/html
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
